from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "featured_iocs.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "ioc_priority_family_bundle_v1.joblib"
SUMMARY_CSV_PATH = PROJECT_ROOT / "reports" / "training_summary_family_v1.csv"
SUMMARY_TXT_PATH = PROJECT_ROOT / "reports" / "training_summary_family_v1.txt"

TARGET_COLUMN = "label"
FAMILY_COLUMN = "indicator_family"

# Family-specific models do not need indicator_family as input because the family is fixed.
FAMILY_CATEGORICAL_FEATURES = [
    "type",
]

# Global fallback model can use indicator_family.
GLOBAL_CATEGORICAL_FEATURES = [
    "type",
    "indicator_family",
]

NUMERICAL_FEATURES = [
    "indicator_length",
    "has_reference",
    "has_context",
    "has_tags",
    "reference_length",
    "context_length",
    "tags_length",
    "tag_count",
    "ioc_age_days",
    "last_seen_days_ago",
    "active_days",
    "seen_last_7d",
    "seen_last_30d",
    "confidence_score",
    "high_keyword_hits",
    "medium_keyword_hits",
    "low_keyword_hits",
    "contains_bracket_ioc",
    "contains_path_or_query",
    "indicator_token_count",
]

MIN_ROWS_TO_TRAIN_FAMILY_MODEL = 50
MIN_ROWS_PER_CLASS_TO_TRAIN_FAMILY_MODEL = 5


def load_training_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Featured dataset not found: {path}\n"
            "Run src/features.py first."
        )

    df = pd.read_csv(path)

    required_columns = (
        [TARGET_COLUMN, FAMILY_COLUMN]
        + list(set(FAMILY_CATEGORICAL_FEATURES + GLOBAL_CATEGORICAL_FEATURES + NUMERICAL_FEATURES))
    )
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    df = df.dropna(subset=[TARGET_COLUMN, FAMILY_COLUMN]).copy()

    if df.empty:
        raise ValueError("Training dataset is empty after dropping missing labels/families.")

    df[FAMILY_COLUMN] = df[FAMILY_COLUMN].fillna("unknown").astype(str).str.strip()

    return df


def build_model(categorical_features: list[str]) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def train_global_fallback_model(df: pd.DataFrame) -> dict[str, object]:
    feature_columns = GLOBAL_CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    model = build_model(GLOBAL_CATEGORICAL_FEATURES)
    model.fit(X, y)

    return {
        "strategy": "model",
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.named_steps["classifier"].classes_),
        "rows": int(len(df)),
    }


def train_family_artifact(family_df: pd.DataFrame) -> dict[str, object]:
    label_counts = family_df[TARGET_COLUMN].value_counts()
    majority_label = label_counts.idxmax()

    rows = int(len(family_df))
    unique_labels = int(label_counts.size)
    min_class_count = int(label_counts.min())

    # Only one class in this family -> no ML model needed.
    if unique_labels == 1:
        return {
            "strategy": "constant",
            "label": majority_label,
            "classes": list(label_counts.index),
            "rows": rows,
            "min_class_count": min_class_count,
        }

    # Tiny or weakly represented family -> use global fallback during scoring.
    if (
        rows < MIN_ROWS_TO_TRAIN_FAMILY_MODEL
        or min_class_count < MIN_ROWS_PER_CLASS_TO_TRAIN_FAMILY_MODEL
    ):
        return {
            "strategy": "fallback_global",
            "label": majority_label,
            "classes": list(label_counts.index),
            "rows": rows,
            "min_class_count": min_class_count,
        }

    feature_columns = FAMILY_CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = family_df[feature_columns].copy()
    y = family_df[TARGET_COLUMN].copy()

    model = build_model(FAMILY_CATEGORICAL_FEATURES)
    model.fit(X, y)

    return {
        "strategy": "model",
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.named_steps["classifier"].classes_),
        "rows": rows,
        "min_class_count": min_class_count,
    }


def main() -> None:
    df = load_training_data(INPUT_PATH)

    all_classes = sorted(df[TARGET_COLUMN].unique().tolist())
    default_label = df[TARGET_COLUMN].value_counts().idxmax()

    print("Training family-based production model bundle")
    print("-" * 50)
    print(f"Rows: {len(df)}")
    print()
    print("Overall label distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print()
    print("Family distribution:")
    print(df[FAMILY_COLUMN].value_counts())
    print()

    global_fallback = train_global_fallback_model(df)

    family_models: dict[str, dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []

    for family, family_df in df.groupby(FAMILY_COLUMN, dropna=False):
        artifact = train_family_artifact(family_df)
        family_models[str(family)] = artifact

        label_counts = family_df[TARGET_COLUMN].value_counts()
        summary_row = {
            "family": str(family),
            "rows": int(len(family_df)),
            "strategy": artifact["strategy"],
            "majority_label": label_counts.idxmax(),
            "unique_labels": int(label_counts.size),
            "min_class_count": int(label_counts.min()),
            "label_distribution": "; ".join(f"{k}={v}" for k, v in label_counts.items()),
        }
        summary_rows.append(summary_row)

    bundle = {
        "bundle_version": "family_v1",
        "target_column": TARGET_COLUMN,
        "family_column": FAMILY_COLUMN,
        "all_classes": all_classes,
        "default_label": default_label,
        "family_categorical_features": FAMILY_CATEGORICAL_FEATURES,
        "global_categorical_features": GLOBAL_CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "global_fallback": global_fallback,
        "family_models": family_models,
    }

    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["rows", "family"], ascending=[False, True])
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    strategy_counts = summary_df["strategy"].value_counts()

    summary_lines = [
        "Family-based Training Summary (v1)",
        "-" * 50,
        f"Rows: {len(df)}",
        f"All classes: {', '.join(all_classes)}",
        f"Default label: {default_label}",
        "",
        "Strategy counts:",
        strategy_counts.to_string(),
        "",
        "Families:",
        summary_df.to_string(index=False),
    ]
    SUMMARY_TXT_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Model bundle saved to: {MODEL_BUNDLE_PATH}")
    print(f"Summary CSV saved to: {SUMMARY_CSV_PATH}")
    print(f"Summary TXT saved to: {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()