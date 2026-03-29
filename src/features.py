from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "labeled" / "labeled_iocs.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "featured_iocs.csv"

IPV4_RE = re.compile(
    r"^(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}$"
)
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
SHA1_RE = re.compile(r"^[a-fA-F0-9]{40}$")
MD5_RE = re.compile(r"^[a-fA-F0-9]{32}$")
URL_RE = re.compile(r"^(?:https?://|hxxps?://)", re.IGNORECASE)
DOMAIN_RE = re.compile(r"^(?!-)(?:[a-zA-Z0-9-]{1,63}\.)+[a-zA-Z]{2,63}$")

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

WEAK_FAMILIES = {"unknown", "other"}
HASH_FAMILIES = {"sha256", "sha1", "md5"}


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def infer_indicator_family(indicator: str) -> str:
    """
    Infer family directly from the indicator value.
    """
    value = normalize_text(indicator)

    if not value:
        return "unknown"

    normalized = value.replace("[.]", ".")

    if URL_RE.match(value):
        return "url"

    if IPV4_RE.match(normalized):
        return "ipv4"

    if SHA256_RE.match(normalized):
        return "sha256"

    if SHA1_RE.match(normalized):
        return "sha1"

    if MD5_RE.match(normalized):
        return "md5"

    if DOMAIN_RE.match(normalized):
        return "domain"

    if normalized.lower().startswith("cve-"):
        return "cve"

    return "other"


def infer_family_from_type(ioc_type: str) -> str:
    """
    Normalize the declared IOC type into a family bucket.
    """
    value = normalize_text(ioc_type).lower()

    mapping = {
        "ip": "ipv4",
        "ipv4": "ipv4",
        "url": "url",
        "uri": "url",
        "link": "url",
        "domain": "domain",
        "fqdn": "domain",
        "sha256": "sha256",
        "sha1": "sha1",
        "md5": "md5",
        "hash": "other",
        "cve": "cve",
    }

    return mapping.get(value, "unknown")


def resolve_final_family(indicator_family_inferred: str, type_family_declared: str) -> str:
    """
    Final family logic:
    - Prefer explicit declared type for known hash families.
    - Prefer indicator-derived family for strong non-hash matches.
    - Fall back cleanly when one side is weak.
    """
    if type_family_declared in HASH_FAMILIES:
        return type_family_declared

    if indicator_family_inferred not in WEAK_FAMILIES:
        return indicator_family_inferred

    if type_family_declared not in WEAK_FAMILIES:
        return type_family_declared

    return indicator_family_inferred


def get_family_resolution_reason(
    indicator_family_inferred: str,
    type_family_declared: str,
) -> str:
    if type_family_declared in HASH_FAMILIES:
        if indicator_family_inferred != type_family_declared:
            return "declared_hash_overrode_inferred"
        return "declared_hash_matched"

    if indicator_family_inferred not in WEAK_FAMILIES:
        if (
            indicator_family_inferred != type_family_declared
            and type_family_declared not in WEAK_FAMILIES
        ):
            return "inferred_overrode_declared"
        return "inferred_family_used"

    if type_family_declared not in WEAK_FAMILIES:
        return "declared_family_used"

    return "weak_family_fallback"


def count_keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for word in keywords if word in lowered)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

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
        if col in featured.columns:
            featured[col] = featured[col].fillna("").astype(str).str.strip()

    featured["first_seen"] = pd.to_datetime(
        featured["first_seen"], errors="coerce", utc=True
    ).dt.tz_localize(None)

    featured["last_seen"] = pd.to_datetime(
        featured["last_seen"], errors="coerce", utc=True
    ).dt.tz_localize(None)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    # Family inference and diagnostics
    featured["indicator_family_inferred"] = featured["indicator"].apply(
        infer_indicator_family
    )
    featured["type_family_declared"] = featured["type"].apply(infer_family_from_type)
    featured["indicator_family"] = featured.apply(
        lambda row: resolve_final_family(
            row["indicator_family_inferred"],
            row["type_family_declared"],
        ),
        axis=1,
    )

    featured["family_mismatch_flag"] = (
        (featured["indicator_family_inferred"] != featured["type_family_declared"])
        & ~featured["indicator_family_inferred"].isin(WEAK_FAMILIES)
        & ~featured["type_family_declared"].isin(WEAK_FAMILIES)
    ).astype(int)

    featured["family_resolution_reason"] = featured.apply(
        lambda row: get_family_resolution_reason(
            row["indicator_family_inferred"],
            row["type_family_declared"],
        ),
        axis=1,
    )

    featured["type_family_known"] = (
        ~featured["type_family_declared"].isin(WEAK_FAMILIES)
    ).astype(int)

    featured["indicator_family_known"] = (
        ~featured["indicator_family_inferred"].isin(WEAK_FAMILIES)
    ).astype(int)

    featured["indicator_length"] = featured["indicator"].str.len()

    # Basic presence / richness
    featured["has_reference"] = featured["reference"].ne("").astype(int)
    featured["has_context"] = featured["context"].ne("").astype(int)
    featured["has_tags"] = featured["tags"].ne("").astype(int)

    featured["reference_length"] = featured["reference"].str.len()
    featured["context_length"] = featured["context"].str.len()
    featured["tags_length"] = featured["tags"].str.len()

    featured["tag_count"] = featured["tags"].apply(
        lambda x: len([tag for tag in str(x).split(",") if tag.strip()])
    )

    # Time features
    featured["ioc_age_days"] = (now - featured["first_seen"]).dt.days
    featured["last_seen_days_ago"] = (now - featured["last_seen"]).dt.days
    featured["active_days"] = (featured["last_seen"] - featured["first_seen"]).dt.days

    featured["ioc_age_days"] = featured["ioc_age_days"].fillna(-1)
    featured["last_seen_days_ago"] = featured["last_seen_days_ago"].fillna(-1)
    featured["active_days"] = featured["active_days"].fillna(0)

    featured["ioc_age_days"] = featured["ioc_age_days"].clip(lower=-1)
    featured["last_seen_days_ago"] = featured["last_seen_days_ago"].clip(lower=-1)
    featured["active_days"] = featured["active_days"].clip(lower=0)

    featured["seen_last_7d"] = (
        (featured["last_seen_days_ago"] >= 0)
        & (featured["last_seen_days_ago"] <= 7)
    ).astype(int)

    featured["seen_last_30d"] = (
        (featured["last_seen_days_ago"] >= 0)
        & (featured["last_seen_days_ago"] <= 30)
    ).astype(int)

    # Confidence as numeric
    confidence_map = {"low": 0, "medium": 1, "high": 2}
    featured["confidence_score"] = (
        featured["confidence"].str.lower().map(confidence_map).fillna(-1).astype(int)
    )

    # Safe text-derived features
    combined_text = (
        featured["type"].fillna("")
        + " "
        + featured["tags"].fillna("")
        + " "
        + featured["reference"].fillna("")
        + " "
        + featured["context"].fillna("")
    ).str.strip()

    featured["high_keyword_hits"] = combined_text.apply(
        lambda x: count_keyword_hits(x, HIGH_KEYWORDS)
    )
    featured["medium_keyword_hits"] = combined_text.apply(
        lambda x: count_keyword_hits(x, MEDIUM_KEYWORDS)
    )
    featured["low_keyword_hits"] = combined_text.apply(
        lambda x: count_keyword_hits(x, LOW_KEYWORDS)
    )

    # Indicator structure
    featured["contains_bracket_ioc"] = (
        featured["indicator"].str.contains(r"\[\.\]", regex=True).astype(int)
    )
    featured["contains_path_or_query"] = (
        featured["indicator"].str.contains(r"[/?=&]", regex=True).astype(int)
    )
    featured["indicator_token_count"] = featured["indicator"].apply(
        lambda x: len([part for part in re.split(r"[^A-Za-z0-9]+", str(x)) if part])
    )

    # Broad category helpers
    featured["is_hash_family"] = featured["indicator_family"].isin(
        HASH_FAMILIES
    ).astype(int)
    featured["is_network_family"] = featured["indicator_family"].isin(
        {"ipv4", "domain", "url"}
    ).astype(int)
    featured["is_vulnerability_family"] = (
        featured["indicator_family"] == "cve"
    ).astype(int)

    # Keep combined_text only for inspection/debugging
    featured["combined_text"] = combined_text

    return featured


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {INPUT_PATH}\n"
            "Run src/label_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    featured_df = build_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(OUTPUT_PATH, index=False)

    print("Feature build complete")
    print("-" * 40)
    print(f"Rows: {len(featured_df)}")
    print(f"Columns: {len(featured_df.columns)}")
    print()

    print("Final indicator families:")
    print(featured_df["indicator_family"].value_counts(dropna=False))
    print()

    print("Declared type families:")
    print(featured_df["type_family_declared"].value_counts(dropna=False))
    print()

    print("Family mismatches:")
    print(featured_df["family_mismatch_flag"].value_counts(dropna=False))
    print()

    print("Family resolution reasons:")
    print(featured_df["family_resolution_reason"].value_counts(dropna=False))
    print()

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()