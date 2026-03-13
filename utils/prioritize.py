"""Variant Prioritization Score — ranks variants 0–100 for clinical actionability.

Scoring logic (additive, capped at 100):
    ACMG classification        0–40 pts
    gnomAD AF rarity           0–20 pts
    SnpEff / VEP impact        0–15 pts
    QUAL score                 0–10 pts
    ClinVar significance       0–15 pts

Final tier:
    80–100 → 🔴 HIGH
    50–79  → 🟠 MEDIUM
    0–49   → 🟢 LOW
"""
from __future__ import annotations
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


def score_variant(row: pd.Series) -> dict:
    """Compute a prioritization score for a single variant row."""
    acmg = ACMG_SCORES.get(str(row.get("acmg_class", "")), 0)
    gnomad = _gnomad_score(row.get("gnomad_af"))
    impact = IMPACT_SCORES.get(str(row.get("annotation_impact", row.get("acmg_path_evidence", ""))).upper().split(",")[0].strip(), 0)
    qual = _qual_score(row.get("quality"))
    clnsig = _clnsig_score(str(row.get("ClinVar Significance", "")))

    total = min(100, acmg + gnomad + impact + qual + clnsig)

    if total >= 80:
        tier = "🔴 HIGH"
    elif total >= 50:
        tier = "🟠 MEDIUM"
    else:
        tier = "🟢 LOW"

    return {
        "priority_score": total,
        "priority_tier": tier,
        "score_breakdown": f"ACMG:{acmg} gnomAD:{gnomad} Impact:{impact} QUAL:{qual} ClinVar:{clnsig}",
    }


def prioritize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add priority_score, priority_tier, score_breakdown columns and sort."""
    if df.empty:
        return df
    scores = df.apply(score_variant, axis=1, result_type="expand")
    result = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    return result.sort_values("priority_score", ascending=False).reset_index(drop=True)
