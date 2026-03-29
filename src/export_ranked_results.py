from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "reports" / "scored_iocs_family_v4.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "exports"

HIGH_SCORE_THRESHOLD = 0.90
MEDIUM_SCORE_THRESHOLD = 0.70
TOP_N_REVIEW = 100


def load_scored_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Scored IOC file not found: {path}\n"
            "Run src/score_iocs.py first."
        )

    df = pd.read_csv(path)

    required_columns = [
        "indicator",
        "indicator_family",
        "source",
        "predicted_label",
        "priority_score",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return df


def sort_for_review(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = []
    ascending = []

    for col, asc in [
        ("priority_rank", False),
        ("priority_score", False),
        ("weak_score", False),
        ("indicator_family", True),
        ("source", True),
    ]:
        if col in df.columns:
            sort_columns.append(col)
            ascending.append(asc)

    if not sort_columns:
        return df.copy()

    return df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)


def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "indicator",
        "indicator_family",
        "type",
        "source",
        "confidence",
        "weak_score",
        "predicted_label",
        "priority_score",
        "priority_rank",
        "scoring_strategy",
        "prior_rule_applied",
        "tags",
        "reference",
        "context",
        "first_seen",
        "last_seen",
    ]

    prob_columns = [col for col in df.columns if col.startswith("prob_")]
    final_columns = [col for col in preferred_columns if col in df.columns] + prob_columns

    return df[final_columns].copy()


def save_family_exports(df: pd.DataFrame, output_dir: Path) -> list[str]:
    created_files: list[str] = []

    if "indicator_family" not in df.columns:
        return created_files

    for family, family_df in df.groupby("indicator_family", dropna=False):
        family_name = str(family).strip().lower() or "unknown"
        family_path = output_dir / f"top_{family_name}_iocs.csv"
        pick_columns(sort_for_review(family_df)).to_csv(family_path, index=False)
        created_files.append(family_path.name)

    return created_files


def build_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append("SwiftIOC ML Scoring Export Summary")
    lines.append("-" * 50)
    lines.append(f"Total scored rows: {len(df)}")
    lines.append("")

    lines.append("Predicted label distribution:")
    lines.append(df["predicted_label"].value_counts().to_string())
    lines.append("")

    if "indicator_family" in df.columns:
        lines.append("Indicator family distribution:")
        lines.append(df["indicator_family"].value_counts().to_string())
        lines.append("")

    if "source" in df.columns:
        lines.append("Top sources overall:")
        lines.append(df["source"].value_counts().head(15).to_string())
        lines.append("")

    high_df = df[df["predicted_label"] == "high_value"].copy()
    medium_df = df[df["predicted_label"] == "medium_value"].copy()
    low_df = df[df["predicted_label"] == "low_value"].copy()

    lines.append(f"High-value rows: {len(high_df)}")
    lines.append(f"Medium-value rows: {len(medium_df)}")
    lines.append(f"Low-value rows: {len(low_df)}")
    lines.append("")

    high_action_df = high_df[high_df["priority_score"] >= HIGH_SCORE_THRESHOLD].copy()
    medium_action_df = medium_df[medium_df["priority_score"] >= MEDIUM_SCORE_THRESHOLD].copy()

    lines.append(
        f"Immediate review candidates (high_value and score >= {HIGH_SCORE_THRESHOLD:.2f}): "
        f"{len(high_action_df)}"
    )
    lines.append(
        f"Queued review candidates (medium_value and score >= {MEDIUM_SCORE_THRESHOLD:.2f}): "
        f"{len(medium_action_df)}"
    )
    lines.append("")

    if not high_df.empty and "source" in high_df.columns:
        lines.append("Top sources in high_value:")
        lines.append(high_df["source"].value_counts().head(15).to_string())
        lines.append("")

    if not high_df.empty:
        preview_cols = [
            col
            for col in [
                "indicator",
                "indicator_family",
                "source",
                "predicted_label",
                "priority_score",
                "scoring_strategy",
                "prior_rule_applied",
            ]
            if col in high_df.columns
        ]
        lines.append("Top 20 high-priority IOCs:")
        lines.append(high_df.sort_values("priority_score", ascending=False)[preview_cols].head(20).to_string(index=False))
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    df = load_scored_data(INPUT_PATH)
    df = sort_for_review(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_df = pick_columns(df)

    high_priority_df = export_df[
        (export_df["predicted_label"] == "high_value")
        & (export_df["priority_score"] >= HIGH_SCORE_THRESHOLD)
    ].copy()

    medium_priority_df = export_df[
        (export_df["predicted_label"] == "medium_value")
        & (export_df["priority_score"] >= MEDIUM_SCORE_THRESHOLD)
    ].copy()

    review_df = export_df.head(TOP_N_REVIEW).copy()

    all_ranked_path = OUTPUT_DIR / "all_ranked_iocs.csv"
    high_priority_path = OUTPUT_DIR / "high_priority_iocs.csv"
    medium_priority_path = OUTPUT_DIR / "medium_priority_iocs.csv"
    review_path = OUTPUT_DIR / f"top_{TOP_N_REVIEW}_for_review.csv"
    summary_path = OUTPUT_DIR / "summary.txt"

    export_df.to_csv(all_ranked_path, index=False)
    high_priority_df.to_csv(high_priority_path, index=False)
    medium_priority_df.to_csv(medium_priority_path, index=False)
    review_df.to_csv(review_path, index=False)

    family_files = save_family_exports(df, OUTPUT_DIR)

    summary_text = build_summary(df)
    summary_path.write_text(summary_text, encoding="utf-8")

    print("Export complete")
    print("-" * 40)
    print(f"Input rows: {len(df)}")
    print(f"All ranked export: {all_ranked_path}")
    print(f"High priority export: {high_priority_path} ({len(high_priority_df)} rows)")
    print(f"Medium priority export: {medium_priority_path} ({len(medium_priority_df)} rows)")
    print(f"Top review export: {review_path} ({len(review_df)} rows)")
    print(f"Summary report: {summary_path}")
    print()

    if family_files:
        print("Family exports created:")
        for name in family_files:
            print(f"- {name}")


if __name__ == "__main__":
    main()