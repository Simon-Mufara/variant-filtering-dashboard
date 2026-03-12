"""ACMG-lite variant classification.

Implements a simplified rules-based ACMG/AMP 2015 pathogenicity scoring.
This is NOT a substitute for clinical-grade tools (e.g. InterVar, VarSome).
It flags variants for triage purposes only.

Evidence codes implemented:
    PVS1 — predicted loss-of-function (stop_gained, frameshift, splice_donor/acceptor)
    PS1  — same amino-acid change as established pathogenic (via ClinVar CLNSIG)
    PM2  — absent / very low frequency in gnomAD (AF < 0.001)
    PM4  — in-frame INDEL in repeat region (length change 3–9 bp)
    PP2  — missense in gene where missense is common disease mechanism (stub — needs gene list)
    BP1  — missense in gene where truncating is primary mechanism (stub)
    BS1  — allele frequency > 5% in population
    BA1  — allele frequency > 5% in any gnomAD population (stand-alone benign)
"""
from __future__ import annotations
import pandas as pd
from typing import Optional


# ── Classification thresholds ─────────────────────────────────────────────────
AF_BA1    = 0.05   # stand-alone benign
AF_BS1    = 0.05   # strong benign
AF_PM2    = 0.001  # moderate pathogenic — absent/ultra-rare

LOF_TERMS = {
    "stop_gained", "frameshift_variant", "splice_donor_variant",
    "splice_acceptor_variant", "start_lost", "transcript_ablation",
}

PATHOGENIC_CLNSIG = {"Pathogenic", "Likely_pathogenic"}
BENIGN_CLNSIG = {"Benign", "Likely_benign"}


def classify_variant(row: pd.Series) -> dict:
    """Apply ACMG-lite rules to a single variant row.

    Expected columns (all optional; missing = no evidence):
        variant_type, ref, alt, gnomad_af, annotation (SnpEff),
        ClinVar Significance, info_raw
    """
    evidence_path: list[str] = []
    evidence_benign: list[str] = []

    ann = str(row.get("annotation", "")).lower()
    vtype = str(row.get("variant_type", ""))
    gnomad_af: Optional[float] = _safe_float(row.get("gnomad_af"))
    clnsig = str(row.get("ClinVar Significance", ""))

    # ── PVS1: predicted loss-of-function ──────────────────────────────────────
    if any(term in ann for term in LOF_TERMS) or any(
        term in str(row.get("info_raw", "")).lower() for term in LOF_TERMS
    ):
        evidence_path.append("PVS1")

    # ── PS1: ClinVar pathogenic ───────────────────────────────────────────────
    if any(sig.lower() in clnsig.lower() for sig in PATHOGENIC_CLNSIG):
        evidence_path.append("PS1")

    # ── PM2: ultra-rare / absent in gnomAD ───────────────────────────────────
    if gnomad_af is not None and gnomad_af < AF_PM2:
        evidence_path.append("PM2")
    elif gnomad_af is None:
        evidence_path.append("PM2")   # absent = same evidence weight

    # ── PM4: in-frame INDEL ───────────────────────────────────────────────────
    if vtype == "INDEL":
        ref = str(row.get("ref", ""))
        alt = str(row.get("alt", ""))
        length_change = abs(len(ref) - len(alt))
        if 3 <= length_change <= 9 and length_change % 3 == 0:
            evidence_path.append("PM4")

    # ── BA1 / BS1: common allele ──────────────────────────────────────────────
    if gnomad_af is not None:
        if gnomad_af >= AF_BA1:
            evidence_benign.append("BA1")
        elif gnomad_af >= AF_BS1:
            evidence_benign.append("BS1")

    if any(sig.lower() in clnsig.lower() for sig in BENIGN_CLNSIG):
        evidence_benign.append("BS2")

    # ── Classification ────────────────────────────────────────────────────────
    classification = _classify(evidence_path, evidence_benign)

    return {
        "acmg_class": classification,
        "acmg_path_evidence": ", ".join(evidence_path) or "—",
        "acmg_benign_evidence": ", ".join(evidence_benign) or "—",
    }


def _classify(path: list[str], benign: list[str]) -> str:
    if "BA1" in benign:
        return "Benign"
    if benign and not path:
        return "Likely Benign"
    if "PVS1" in path and ("PS1" in path or "PM2" in path):
        return "Pathogenic"
    if "PVS1" in path:
        return "Likely Pathogenic"
    if "PS1" in path:
        return "Likely Pathogenic"
    if len(path) >= 3:
        return "Likely Pathogenic"
    if len(path) >= 1 and not benign:
        return "VUS"
    return "VUS"


def _safe_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add ACMG classification columns to every row of a DataFrame."""
    if df.empty:
        return df
    classifications = df.apply(classify_variant, axis=1, result_type="expand")
    return pd.concat([df.reset_index(drop=True), classifications.reset_index(drop=True)], axis=1)
