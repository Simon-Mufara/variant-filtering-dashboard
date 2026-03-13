"""Variant Prioritization Score — ranks variants 0–100 for clinical actionability.

Scoring logic (additive, capped at 100):
    ACMG classification        0–40 pts
    gnomAD AF rarity           0–20 pts
    SnpEff / VEP impact        0–15 pts
    QUAL score                 0–10 pts
    ClinVar significance       0–15 pts
    Predictor scores bonus     0–10 pts  (CADD ≥20, REVEL ≥0.75, AM ≥0.75)

Final tier:
    80–100 → 🔴 HIGH
    50–79  → 🟠 MEDIUM
    0–49   → 🟢 LOW
"""
from __future__ import annotations
import re
import pandas as pd


ACMG_SCORES = {
    "Pathogenic": 40,
    "Likely Pathogenic": 30,
    "VUS": 15,
    "Likely Benign": 5,
    "Benign": 0,
}

IMPACT_SCORES = {
    "HIGH": 15,
    "MODERATE": 8,
    "LOW": 3,
    "MODIFIER": 1,
}

CLNSIG_SCORES = {
    "pathogenic": 15,
    "likely_pathogenic": 12,
    "vus": 5,
    "uncertain_significance": 5,
    "likely_benign": 2,
    "benign": 0,
}

# Regex to pull CLNSIG from raw INFO string
_CLNSIG_RE = re.compile(r"CLNSIG=([^;]+)", re.IGNORECASE)
# Regex to pull impact from SnpEff ANN field (3rd pipe-delimited field)
_ANN_IMPACT_RE = re.compile(r"ANN=[^|]*\|[^|]*\|([^|]+)\|")


def _gnomad_score(af) -> int:
    """Rarity score from gnomAD AF — rarer = more points."""
    try:
        af = float(af)
    except (TypeError, ValueError):
        return 15  # absent = rare = high score
    if af < 0.0001:
        return 20
    if af < 0.001:
        return 15
    if af < 0.01:
        return 10
    if af < 0.05:
        return 5
    return 0


def _qual_score(qual) -> int:
    try:
        q = float(qual)
    except (TypeError, ValueError):
        return 5
    if q >= 200:
        return 10
    if q >= 100:
        return 8
    if q >= 50:
        return 5
    if q >= 20:
        return 3
    return 0


def _clnsig_score(clnsig: str) -> int:
    if not clnsig:
        return 0
    low = clnsig.lower()
    for key, pts in CLNSIG_SCORES.items():
        if key in low:
            return pts
    return 0


def _predictor_bonus(row: pd.Series) -> int:
    """Bonus points from CADD, REVEL, and AlphaMissense scores (0–10)."""
    bonus = 0
    cadd = row.get("cadd_phred")
    revel = row.get("revel_score")
    am = row.get("alphamissense_score")
    try:
        if cadd is not None and float(cadd) >= 20:
            bonus += 3
        if cadd is not None and float(cadd) >= 30:
            bonus += 2  # extra for very high CADD
    except (TypeError, ValueError):
        pass
    try:
        if revel is not None and float(revel) >= 0.5:
            bonus += 2
        if revel is not None and float(revel) >= 0.75:
            bonus += 1
    except (TypeError, ValueError):
        pass
    try:
        if am is not None and float(am) >= 0.5:
            bonus += 2
        if am is not None and float(am) >= 0.75:
            bonus += 1
    except (TypeError, ValueError):
        pass
    return min(10, bonus)


def _extract_clnsig(row: pd.Series) -> str:
    """Get ClinVar significance from explicit column or INFO string."""
    # Prefer already-parsed column
    explicit = str(row.get("ClinVar Significance", "")).strip()
    if explicit and explicit not in ("", "Unknown", "nan"):
        return explicit
    # Fall back to parsing info_raw
    info = str(row.get("info_raw", ""))
    m = _CLNSIG_RE.search(info)
    return m.group(1).strip() if m else ""


def _extract_impact(row: pd.Series) -> str:
    """Get variant impact from explicit column or ANN INFO field."""
    explicit = str(row.get("annotation_impact", row.get("vep_impact", ""))).strip()
    if explicit and explicit.upper() in IMPACT_SCORES:
        return explicit.upper()
    # Parse from ANN= field in info_raw
    info = str(row.get("info_raw", ""))
    m = _ANN_IMPACT_RE.search(info)
    if m:
        return m.group(1).strip().upper()
    return ""


def score_variant(row: pd.Series) -> dict:
    """Compute a prioritization score for a single variant row."""
    acmg = ACMG_SCORES.get(str(row.get("acmg_class", "")), 0)
    gnomad = _gnomad_score(row.get("gnomad_af"))
    impact_key = _extract_impact(row)
    impact = IMPACT_SCORES.get(impact_key, 0)
    qual = _qual_score(row.get("quality"))
    clnsig = _clnsig_score(_extract_clnsig(row))
    pred_bonus = _predictor_bonus(row)

    total = min(100, acmg + gnomad + impact + qual + clnsig + pred_bonus)

    if total >= 80:
        tier = "🔴 HIGH"
    elif total >= 50:
        tier = "🟠 MEDIUM"
    else:
        tier = "🟢 LOW"

    return {
        "priority_score": total,
        "priority_tier": tier,
        "score_breakdown": (
            f"ACMG:{acmg} gnomAD:{gnomad} Impact:{impact} "
            f"QUAL:{qual} ClinVar:{clnsig} Predictors:{pred_bonus}"
        ),
    }


def prioritize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add priority_score, priority_tier, score_breakdown columns and sort."""
    if df.empty:
        return df
    scores = df.apply(score_variant, axis=1, result_type="expand")
    result = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    return result.sort_values("priority_score", ascending=False).reset_index(drop=True)
