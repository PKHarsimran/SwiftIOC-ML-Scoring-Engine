from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import build_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "latest.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "ioc_priority_family_bundle_v1.joblib"
SCORED_OUTPUT_PATH = PROJECT_ROOT / "reports" / "scored_iocs_family_v4.csv"


def load_input_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Copy the newest SwiftIOC export into data/raw/latest.csv first."
        )

    return pd.read_csv(path)


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


def normalize_probability_block(
    df: pd.DataFrame,
    mask: pd.Series,
    prob_columns: list[str],
) -> None:
    if not mask.any():
        return

    block = df.loc[mask, prob_columns].astype(float).clip(lower=0.0)
    row_sums = block.sum(axis=1)

    zero_sum_mask = row_sums <= 0
    nonzero_mask = ~zero_sum_mask

    if nonzero_mask.any():
        block.loc[nonzero_mask] = block.loc[nonzero_mask].div(
            row_sums.loc[nonzero_mask],
            axis=0,
        )

    if zero_sum_mask.any():
        uniform = 1.0 / len(prob_columns)
        block.loc[zero_sum_mask, :] = uniform

    df.loc[mask, prob_columns] = block


def append_strategy(
    df: pd.DataFrame,
    mask: pd.Series,
    suffix: str,
) -> None:
    if not mask.any():
        return

    current = df.loc[mask, "scoring_strategy"].fillna("").astype(str).str.strip()
    updated = current.where(current.eq(""), current + "|") + suffix
    df.loc[mask, "scoring_strategy"] = updated


def set_hard_override(
    df: pd.DataFrame,
    mask: pd.Series,
    label: str,
    all_classes: list[str],
    strategy_suffix: str,
    rule_name: str,
) -> None:
    if not mask.any():
        return

    for class_name in all_classes:
        df.loc[mask, f"prob_{class_name}"] = 1.0 if class_name == label else 0.0

    df.loc[mask, "predicted_label"] = label
    append_strategy(df, mask, strategy_suffix)
    df.loc[mask, "prior_rule_applied"] = rule_name


def apply_source_rules_vectorized(
    scored_df: pd.DataFrame,
    all_classes: list[str],
) -> pd.DataFrame:
    """
    Apply source-aware production rules without row-wise apply.
    """
    prob_columns = [f"prob_{class_name}" for class_name in all_classes]

    source_norm = scored_df["source"].fillna("").astype(str).str.strip().str.lower()
    family_norm = scored_df["indicator_family"].fillna("").astype(str).str.strip().str.lower()

    is_url = family_norm.eq("url")
    is_ipv4 = family_norm.eq("ipv4")

    # Exact source-group overrides
    openphish_mask = is_url & source_norm.eq("openphish_feed")
    tor_mask = is_ipv4 & source_norm.eq("tor_exit_nodes")
    feodo_mask = is_ipv4 & source_norm.eq("feodo_ipblocklist")

    # Soft medium bias for scanner/blocklist-style ipv4 feeds.
    # Match names inside comma-separated source strings.
    scanner_pattern = (
        r"(?:^|,\s*)(?:greensnow_blocklist|ci_army_list|blocklist_de_ssh)(?:\s*,|$)"
    )
    scanner_mask = (
        is_ipv4
        & source_norm.str.contains(scanner_pattern, regex=True, na=False)
        & ~tor_mask
        & ~feodo_mask
    )

    # Hard overrides first
    set_hard_override(
        df=scored_df,
        mask=openphish_mask,
        label="high_value",
        all_classes=all_classes,
        strategy_suffix="url_override",
        rule_name="openphish_feed=>high_override",
    )

    set_hard_override(
        df=scored_df,
        mask=tor_mask,
        label="low_value",
        all_classes=all_classes,
        strategy_suffix="ipv4_override",
        rule_name="tor_exit_nodes=>low_override",
    )

    set_hard_override(
        df=scored_df,
        mask=feodo_mask,
        label="high_value",
        all_classes=all_classes,
        strategy_suffix="ipv4_override",
        rule_name="feodo_ipblocklist=>high_override",
    )

    # Soft bias for scanner/blocklist ipv4 feeds
    if scanner_mask.any():
        if "prob_medium_value" in scored_df.columns:
            scored_df.loc[scanner_mask, "prob_medium_value"] = (
                scored_df.loc[scanner_mask, "prob_medium_value"].astype(float) + 0.10
            )
        if "prob_low_value" in scored_df.columns:
            scored_df.loc[scanner_mask, "prob_low_value"] = (
                scored_df.loc[scanner_mask, "prob_low_value"].astype(float) - 0.03
            )
        if "prob_high_value" in scored_df.columns:
            scored_df.loc[scanner_mask, "prob_high_value"] = (
                scored_df.loc[scanner_mask, "prob_high_value"].astype(float) - 0.03
            )

        normalize_probability_block(scored_df, scanner_mask, prob_columns)
        append_strategy(scored_df, scanner_mask, "ipv4_prior")
        scored_df.loc[scanner_mask, "prior_rule_applied"] = "scanner_blocklist=>medium_bias"

    # Any rows not touched by rules keep blank prior marker
    untouched_mask = scored_df["prior_rule_applied"].isna() | scored_df["prior_rule_applied"].eq("")
    scored_df.loc[untouched_mask, "prior_rule_applied"] = scored_df.loc[
        untouched_mask, "prior_rule_applied"
    ].fillna("")

    return scored_df


def main() -> None:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_BUNDLE_PATH}\n"
            "Run src/train_model.py first."
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)

    family_column = bundle["family_column"]
    all_classes = list(bundle["all_classes"])
    default_label = bundle["default_label"]
    family_models: dict[str, dict[str, object]] = bundle["family_models"]
    global_fallback: dict[str, object] = bundle["global_fallback"]

    raw_df = load_input_data(RAW_INPUT_PATH)
    featured_df = build_features(raw_df).copy()

    if family_column not in featured_df.columns:
        raise ValueError(f"Missing required family column: {family_column}")

    featured_df[family_column] = (
        featured_df[family_column].fillna("unknown").astype(str).str.strip()
    )

    scored_df = featured_df.copy()
    scored_df["predicted_label"] = ""
    scored_df["scoring_strategy"] = ""
    scored_df["prior_rule_applied"] = ""

    for class_name in all_classes:
        scored_df[f"prob_{class_name}"] = 0.0

    # Family routing
    for family, subset_df in scored_df.groupby(family_column, dropna=False, sort=False):
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

    # Vectorized source-aware rules
    scored_df = apply_source_rules_vectorized(scored_df, all_classes)

    # Final rank + score, vectorized
    prob_columns = [f"prob_{class_name}" for class_name in all_classes]
    best_prob_col = scored_df[prob_columns].idxmax(axis=1)

    scored_df["predicted_label"] = best_prob_col.str.replace("prob_", "", regex=False)
    scored_df["priority_score"] = scored_df[prob_columns].max(axis=1)

    label_rank = {
        "high_value": 3,
        "medium_value": 2,
        "low_value": 1,
    }
    scored_df["priority_rank"] = (
        scored_df["predicted_label"].map(label_rank).fillna(0).astype(int)
    )

    sort_columns = ["priority_rank", "priority_score"]
    ascending = [False, False]

    if "weak_score" in scored_df.columns:
        sort_columns.append("weak_score")
        ascending.append(False)

    scored_df = scored_df.sort_values(
        by=sort_columns,
        ascending=ascending,
    ).reset_index(drop=True)

    output_columns = [
        "indicator",
        "type",
        "source",
        "indicator_family",
        "confidence",
        "tags",
        "reference",
        "context",
        "first_seen",
        "last_seen",
        "weak_score",
        "predicted_label",
        "priority_score",
        "priority_rank",
        "scoring_strategy",
        "prior_rule_applied",
    ]

    final_columns = [col for col in output_columns if col in scored_df.columns] + prob_columns
    final_df = scored_df[final_columns].copy()

    SCORED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(SCORED_OUTPUT_PATH, index=False)

    print("Family-based scoring complete (v4 with source overrides)")
    print("-" * 50)
    print(f"Rows scored: {len(final_df)}")
    print(f"Saved scored IOC file to: {SCORED_OUTPUT_PATH}")
    print()
    print("Top 10 scored IOCs:")
    preview_cols = [
        "indicator",
        "indicator_family",
        "type",
        "source",
        "predicted_label",
        "priority_score",
        "scoring_strategy",
        "prior_rule_applied",
    ]
    preview_cols = [col for col in preview_cols if col in final_df.columns]
    print(final_df[preview_cols].head(10))
    print()
    print("Scoring strategy usage:")
    print(final_df["scoring_strategy"].value_counts())
    print()
    print("Prior rule usage:")
    print(final_df["prior_rule_applied"].value_counts().head(20))


if __name__ == "__main__":
    main()