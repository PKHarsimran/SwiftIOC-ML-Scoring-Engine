
from __future__ import annotations

import pandas as pd

TEXT_COLUMNS = [
    "indicator",
    "type",
    "source",
    "confidence",
    "tags",
    "reference",
    "context",
]


def preprocess_ioc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize IOC data for downstream ML tasks."""
    cleaned = df.copy()

    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    if "type" in cleaned.columns:
        cleaned["type"] = cleaned["type"].str.lower()

    if "source" in cleaned.columns:
        cleaned["source"] = cleaned["source"].str.lower()

    if "confidence" in cleaned.columns:
        cleaned["confidence"] = cleaned["confidence"].str.lower()

    if "first_seen" in cleaned.columns:
        cleaned["first_seen"] = pd.to_datetime(cleaned["first_seen"], errors="coerce")

    if "last_seen" in cleaned.columns:
        cleaned["last_seen"] = pd.to_datetime(cleaned["last_seen"], errors="coerce")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned