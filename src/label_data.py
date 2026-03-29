from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "processed_iocs.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled" / "labeled_iocs.csv"

HIGH_KEYWORDS = {
    "malware",
    "phishing",
    "c2",
    "ransomware",
    "botnet",
    "trojan",
    "stealer",
    "exploit",
    "loader",
    "backdoor",
    "credential",
    "known exploited",
}

MEDIUM_KEYWORDS = {
    "suspicious",
    "abuse",
    "scanner",
    "spam",
    "proxy",
    "bruteforce",
    "scan",
    "anonymous",
}

LOW_KEYWORDS = {
    "benign",
    "allowlist",
    "test",
    "internal",
    "false positive",
    "fp",
}


def normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def infer_indicator_family(indicator: str) -> str:
    value = normalize_text(indicator).replace("[.]", ".")

    if not value:
        return "unknown"

    if value.startswith("cve-"):
        return "cve"

    if value.startswith(("http://", "https://", "hxxp://", "hxxps://")):
        return "url"

    hex_chars = set("0123456789abcdef")
    if len(value) == 64 and all(ch in hex_chars for ch in value):
        return "sha256"
    if len(value) == 40 and all(ch in hex_chars for ch in value):
        return "sha1"
    if len(value) == 32 and all(ch in hex_chars for ch in value):
        return "md5"

    parts = value.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return "ipv4"

    return "other"


def count_keyword_hits(text: str, keywords: set[str]) -> int:
    return sum(1 for word in keywords if word in text)


def compute_weak_score(row: pd.Series) -> int:
    confidence = normalize_text(row.get("confidence", ""))
    tags = normalize_text(row.get("tags", ""))
    context = normalize_text(row.get("context", ""))
    reference = normalize_text(row.get("reference", ""))
    indicator = normalize_text(row.get("indicator", ""))
    ioc_type = normalize_text(row.get("type", ""))

    indicator_family = infer_indicator_family(indicator)

    # IMPORTANT: do not include source here
    combined_text = " ".join(
        part for part in [tags, context, reference, indicator, ioc_type] if part
    )

    score = 0

    # Confidence
    if confidence == "high":
        score += 2
    elif confidence == "medium":
        score += 1

    # Indicator family / type
    if indicator_family == "cve":
        score += 4
    elif indicator_family in {"sha256", "sha1", "md5"}:
        score += 2
    elif indicator_family == "url":
        score += 2
    elif indicator_family == "ipv4":
        score += 0

    if ioc_type == "cve":
        score += 2
    elif ioc_type in {"sha256", "sha1", "md5", "url"}:
        score += 1

    # Context richness
    if reference:
        score += 1
    if context:
        score += 1

    # Tags help a little, but should not dominate
    if tags:
        score += 1

    # Keyword scoring
    high_hits = count_keyword_hits(combined_text, HIGH_KEYWORDS)
    medium_hits = count_keyword_hits(combined_text, MEDIUM_KEYWORDS)
    low_hits = count_keyword_hits(combined_text, LOW_KEYWORDS)

    score += min(high_hits, 2) * 2
    score += min(medium_hits, 2)
    score -= min(low_hits, 2) * 2

    return max(score, 0)


def score_to_label(score: int) -> str:
    if score >= 8:
        return "high_value"
    if score >= 4:
        return "medium_value"
    return "low_value"


def label_ioc_data(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["weak_score"] = labeled.apply(compute_weak_score, axis=1)
    labeled["label"] = labeled["weak_score"].apply(score_to_label)
    return labeled


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_PATH}\n"
            "Run src/load_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    labeled_df = label_ioc_data(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(OUTPUT_PATH, index=False)

    print("Weak score distribution:")
    print(labeled_df["weak_score"].value_counts().sort_index())
    print()
    print("Label distribution:")
    print(labeled_df["label"].value_counts())
    print()
    print("Label %:")
    print((labeled_df["label"].value_counts(normalize=True) * 100).round(2))
    print(f"\nLabeled dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()