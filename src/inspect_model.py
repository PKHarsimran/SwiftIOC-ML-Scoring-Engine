from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ioc_priority_model_v2.joblib"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "feature_importance_v2.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run src/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = list(preprocessor.get_feature_names_out())
    classes = list(classifier.classes_)
    coefs = classifier.coef_

    print(f"Model classes: {classes}")
    print(f"Coefficient shape: {coefs.shape}")
    print()

    # Binary classification: sklearn stores one row only
    if len(classes) == 2 and coefs.shape[0] == 1:
        positive_class = classes[1]
        negative_class = classes[0]

        coef_df = pd.DataFrame(
            {
                "feature_name": feature_names,
                f"weight_for_{positive_class}": coefs[0],
                f"weight_for_{negative_class}": -coefs[0],
            }
        )

        coef_df[f"abs_weight_for_{positive_class}"] = coef_df[
            f"weight_for_{positive_class}"
        ].abs()
        coef_df[f"abs_weight_for_{negative_class}"] = coef_df[
            f"weight_for_{negative_class}"
        ].abs()

        coef_df["max_abs_weight"] = coef_df[
            [
                f"abs_weight_for_{positive_class}",
                f"abs_weight_for_{negative_class}",
            ]
        ].max(axis=1)

    else:
        coef_df = pd.DataFrame(coefs.T, index=feature_names, columns=classes).reset_index()
        coef_df = coef_df.rename(columns={"index": "feature_name"})

        abs_cols = []
        for class_name in classes:
            abs_col = f"{class_name}_abs"
            coef_df[abs_col] = coef_df[class_name].abs()
            abs_cols.append(abs_col)

        coef_df["max_abs_weight"] = coef_df[abs_cols].max(axis=1)

    coef_df = coef_df.sort_values("max_abs_weight", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(OUTPUT_PATH, index=False)

    print("Top 25 strongest features:")
    print(coef_df.head(25).to_string(index=False))
    print()
    print(f"Saved feature importance to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()