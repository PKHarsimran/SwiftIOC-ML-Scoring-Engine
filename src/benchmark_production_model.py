from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from train_model import (
    FAMILY_COLUMN,
    TARGET_COLUMN,
    load_training_data,
    train_family_artifact,
    train_global_fallback_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "featured_iocs.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "benchmark_family_by_source_group_v3.csv"

ALL_CLASSES = ["high_value", "medium_value", "low_value"]


def normalize_source_group(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return "unknown"

    return "|".join(sorted(set(parts)))


def normalize_probabilities(row: pd.Series, prob_columns: list[str]) -> pd.Series:
    values = row[prob_columns].astype(float).clip(lower=0.0)
    total = values.sum()

    if total <= 0:
        uniform = 1.0 / len(prob_columns)
        for col in prob_columns:
            row[col] = uniform
        return row

    for col in prob_columns:
        row[col] = float(values[col] / total)

    return row


def apply_ipv4_source_prior(row: pd.Series, all_classes: list[str]) -> pd.Series:
    """
    Mirror the current production ipv4 rules exactly:
    - hard override pure tor_exit_nodes -> low_value
    - hard override pure feodo_ipblocklist -> high_value
    - soft medium bias for scanner/blocklist ipv4 feeds
    """
    if str(row.get("indicator_family", "")).strip().lower() != "ipv4":
        row["prior_rule_applied"] = str(row.get("prior_rule_applied", ""))
        return row

    source = str(row.get("source", "")).strip().lower()
    source_parts = {part.strip() for part in source.split(",") if part.strip()}
    prob_columns = [f"prob_{class_name}" for class_name in all_classes]

    for col in prob_columns:
        if col not in row.index:
            row[col] = 0.0

    current_strategy = str(row.get("scoring_strategy", "")).strip()
    applied_rules: list[str] = []

    def set_hard_override(label: str, rule_name: str) -> pd.Series:
        for class_name in all_classes:
            row[f"prob_{class_name}"] = 1.0 if class_name == label else 0.0

        row["predicted_label"] = label
        row["scoring_strategy"] = f"{current_strategy}|ipv4_override".strip("|")
        row["prior_rule_applied"] = rule_name
        return row

    def add_to_prob(label: str, amount: float) -> None:
        col = f"prob_{label}"
        if col in row.index:
            row[col] = float(row[col]) + amount

    if source_parts == {"tor_exit_nodes"}:
        return set_hard_override("low_value", "tor_exit_nodes=>low_override")

    if source_parts == {"feodo_ipblocklist"}:
        return set_hard_override("high_value", "feodo_ipblocklist=>high_override")

    medium_ipv4_sources = {
        "greensnow_blocklist",
        "ci_army_list",
        "blocklist_de_ssh",
    }
    if source_parts & medium_ipv4_sources:
        add_to_prob("medium_value", 0.10)
        add_to_prob("low_value", -0.03)
        add_to_prob("high_value", -0.03)
        applied_rules.append("scanner_blocklist=>medium_bias")

    row = normalize_probabilities(row, prob_columns)

    if applied_rules:
        row["scoring_strategy"] = f"{current_strategy}|ipv4_prior".strip("|")
        row["prior_rule_applied"] = ";".join(applied_rules)
    else:
        row["prior_rule_applied"] = ""

    return row


def rank_from_probabilities(row: pd.Series, all_classes: list[str]) -> str:
    prob_columns = [f"prob_{class_name}" for class_name in all_classes]
    best_col = max(prob_columns, key=lambda col: float(row[col]))
    return best_col.replace("prob_", "", 1)


def build_family_bundle(train_df: pd.DataFrame) -> dict[str, object]:
    default_label = train_df[TARGET_COLUMN].value_counts().idxmax()

    global_fallback = train_global_fallback_model(train_df)

    family_models: dict[str, dict[str, object]] = {}
    for family, family_df in train_df.groupby(FAMILY_COLUMN, dropna=False):
        family_models[str(family)] = train_family_artifact(family_df)

    return {
        "all_classes": ALL_CLASSES,
        "default_label": default_label,
        "global_fallback": global_fallback,
        "family_models": family_models,
    }


def apply_constant_prediction(
    scored_df: pd.DataFrame,
    indices: pd.Index,
    label: str,
    all_classes: list[str],
    strategy_name: str,
) -> None:
    scored_df.loc[indices, "predicted_label"] = label
    scored_df.loc[indices, "scoring_strategy"] = strategy_name

    for class_name in all_classes:
        scored_df.loc[indices, f"prob_{class_name}"] = 1.0 if class_name == label else 0.0


def apply_model_prediction(
    scored_df: pd.DataFrame,
    indices: pd.Index,
    subset_df: pd.DataFrame,
    model_entry: dict[str, object],
    all_classes: list[str],
    strategy_name: str,
) -> None:
    model = model_entry["model"]
    feature_columns = model_entry["feature_columns"]

    missing_columns = [col for col in feature_columns if col not in subset_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required feature columns for scoring: {', '.join(missing_columns)}"
        )

    X = subset_df[feature_columns].copy()
    predicted = model.predict(X)

    scored_df.loc[indices, "predicted_label"] = predicted
    scored_df.loc[indices, "scoring_strategy"] = strategy_name

    for class_name in all_classes:
        scored_df.loc[indices, f"prob_{class_name}"] = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        model_classes = list(model.named_steps["classifier"].classes_)

        for i, class_name in enumerate(model_classes):
            scored_df.loc[indices, f"prob_{class_name}"] = probabilities[:, i]
    else:
        for row_idx, predicted_label in zip(indices, predicted):
            scored_df.loc[row_idx, f"prob_{predicted_label}"] = 1.0


def score_with_family_bundle(bundle: dict[str, object], test_df: pd.DataFrame) -> pd.DataFrame:
    all_classes = list(bundle["all_classes"])
    default_label = bundle["default_label"]
    family_models: dict[str, dict[str, object]] = bundle["family_models"]
    global_fallback: dict[str, object] = bundle["global_fallback"]

    scored_df = test_df.copy()
    scored_df["predicted_label"] = ""
    scored_df["scoring_strategy"] = ""
    scored_df["prior_rule_applied"] = ""

    for class_name in all_classes:
        scored_df[f"prob_{class_name}"] = 0.0

    for family, subset_df in scored_df.groupby(FAMILY_COLUMN, dropna=False):
        family = str(family)
        indices = subset_df.index
        entry = family_models.get(family)

        if entry is None:
            apply_model_prediction(
                scored_df=scored_df,
                indices=indices,
                subset_df=subset_df,
                model_entry=global_fallback,
                all_classes=all_classes,
                strategy_name="global_fallback_unseen_family",
            )
            continue

        strategy = entry["strategy"]

        if strategy == "model":
            apply_model_prediction(
                scored_df=scored_df,
                indices=indices,
                subset_df=subset_df,
                model_entry=entry,
                all_classes=all_classes,
                strategy_name=f"family_model:{family}",
            )
        elif strategy == "constant":
            apply_constant_prediction(
                scored_df=scored_df,
                indices=indices,
                label=entry["label"],
                all_classes=all_classes,
                strategy_name=f"family_constant:{family}",
            )
        elif strategy == "fallback_global":
            apply_model_prediction(
                scored_df=scored_df,
                indices=indices,
                subset_df=subset_df,
                model_entry=global_fallback,
                all_classes=all_classes,
                strategy_name=f"global_fallback:{family}",
            )
        else:
            apply_constant_prediction(
                scored_df=scored_df,
                indices=indices,
                label=default_label,
                all_classes=all_classes,
                strategy_name=f"default_label:{family}",
            )

    scored_df = scored_df.apply(lambda row: apply_ipv4_source_prior(row, all_classes), axis=1)
    scored_df["predicted_label"] = scored_df.apply(
        lambda row: rank_from_probabilities(row, all_classes),
        axis=1,
    )
    return scored_df


def main() -> None:
    df = load_training_data(INPUT_PATH)
    df["source_group"] = df["source"].apply(normalize_source_group)

    unique_groups = sorted(df["source_group"].unique().tolist())
    results: list[dict[str, object]] = []

    print("Running family-based benchmark with ipv4 overrides...")
    print(f"Total rows: {len(df)}")
    print(f"Unique source groups: {len(unique_groups)}")
    print()

    for fold_num, held_out_group in enumerate(unique_groups, start=1):
        test_df = df[df["source_group"] == held_out_group].copy()
        train_df = df[df["source_group"] != held_out_group].copy()

        bundle = build_family_bundle(train_df)
        scored_test_df = score_with_family_bundle(bundle, test_df)

        y_true = scored_test_df[TARGET_COLUMN].copy()
        y_pred = scored_test_df["predicted_label"].copy()

        prior_counts = scored_test_df["prior_rule_applied"].value_counts()
        row = {
            "fold": fold_num,
            "held_out_source_group": held_out_group,
            "test_rows": len(test_df),
            "train_rows": len(train_df),
            "actual_high_value": int((y_true == "high_value").sum()),
            "actual_medium_value": int((y_true == "medium_value").sum()),
            "actual_low_value": int((y_true == "low_value").sum()),
            "pred_high_value": int((y_pred == "high_value").sum()),
            "pred_medium_value": int((y_pred == "medium_value").sum()),
            "pred_low_value": int((y_pred == "low_value").sum()),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "prior_nonempty_rows": int((scored_test_df["prior_rule_applied"] != "").sum()),
            "top_prior_rule": prior_counts.index[0] if not prior_counts.empty else "",
        }
        results.append(row)

        print(
            f"[Fold {fold_num:02d}] "
            f"group={held_out_group} | "
            f"rows={len(test_df)} | "
            f"macro_f1={row['f1_macro']:.4f} | "
            f"weighted_f1={row['f1_weighted']:.4f} | "
            f"prior_rows={row['prior_nonempty_rows']}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_macro", "test_rows"],
        ascending=[True, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Family benchmark with ipv4 overrides complete")
    print("-" * 40)
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("Average scores:")
    print(results_df[["precision_macro", "recall_macro", "f1_macro", "f1_weighted"]].mean())
    print()
    print("Worst held-out groups:")
    print(
        results_df[
            [
                "held_out_source_group",
                "test_rows",
                "actual_high_value",
                "actual_medium_value",
                "actual_low_value",
                "f1_macro",
                "f1_weighted",
                "prior_nonempty_rows",
                "top_prior_rule",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()