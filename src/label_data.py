from __future__ import annotations

from pathlib import Path
import pandas as pd

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


def load_ioc_data(input_path: str | Path) -> pd.DataFrame:
    """Load IOC data exported from SwiftIOC."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Copy latest.csv from the SwiftIOC repo into data/raw/latest.csv first."
        )

    df = pd.read_csv(path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    return df


def clean_ioc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleanup for IOC datasets."""
    cleaned = df.copy()

    text_columns = [
        "indicator",
        "type",
        "source",
        "confidence",
        "tags",
        "reference",
        "context",
    ]

    for col in text_columns:
        cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    cleaned["first_seen"] = pd.to_datetime(cleaned["first_seen"], errors="coerce")
    cleaned["last_seen"] = pd.to_datetime(cleaned["last_seen"], errors="coerce")

    cleaned["type"] = cleaned["type"].str.lower()
    cleaned["source"] = cleaned["source"].str.lower()
    cleaned["confidence"] = cleaned["confidence"].str.lower()

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print("Dataset summary")
    print("-" * 40)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("IOC types:")
    print(df["type"].value_counts(dropna=False).head(10))
    print()

    print("Top sources:")
    print(df["source"].value_counts(dropna=False).head(10))
    print()


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned IOC data to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    input_path = Path("data/raw/latest.csv")
    output_path = Path("data/processed/processed_iocs.csv")

    df = load_ioc_data(input_path)
    cleaned_df = clean_ioc_data(df)
    save_processed_data(cleaned_df, output_path)
    print_summary(cleaned_df)

    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
