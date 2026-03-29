from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocess import preprocess_ioc_data

REQUIRED_COLUMNS = [
    "indicator",
    "type",
    "source",
    "first_seen",
    "last_seen",
    "confidence",
    "tags",
    "reference",
    "context",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "latest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_iocs.csv"


def load_ioc_data(input_path: str | Path) -> pd.DataFrame:
    """Load IOC data exported from SwiftIOC and validate required columns."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Copy latest.csv from the SwiftIOC repo into data/raw/latest.csv first."
        )

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    df.columns = [str(col).strip() for col in df.columns]

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(missing_columns)}\n"
            f"Available columns: {', '.join(df.columns)}"
        )

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the cleaned dataset."""
    print("Dataset summary")
    print("-" * 40)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    if "type" in df.columns:
        print("IOC types:")
        print(df["type"].value_counts(dropna=False).head(10))
        print()

    if "source" in df.columns:
        print("Top sources:")
        print(df["source"].value_counts(dropna=False).head(10))
        print()

    print("Missing values:")
    print(df[REQUIRED_COLUMNS].isna().sum())
    print()


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned IOC data to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df = load_ioc_data(INPUT_PATH)
    cleaned_df = preprocess_ioc_data(df)

    save_processed_data(cleaned_df, OUTPUT_PATH)
    print_summary(cleaned_df)

    print(f"Processed dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()