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

# African contextualisation thresholds
AFRICAN_PM2_THRESHOLD = 0.001
AFRICAN_BA1_THRESHOLD = 0.05
AFRICAN_DISCREPANCY_RATIO = 10  # ratio of African to overall AF
AFRICAN_MIN_AF = 0.005  # minimum African AF to flag


def classify_variant(row: pd.Series) -> dict:
    """Apply ACMG-lite rules to a single variant row.

    Expected columns (all optional; missing = no evidence):
        variant_type, ref, alt, gnomad_af, gnomad_af_afr, annotation (SnpEff),
        ClinVar Significance, info_raw
    """
    evidence_path: list[str] = []
    evidence_benign: list[str] = []
    african_context = None

    ann = str(row.get("annotation", "")).lower()
    vtype = str(row.get("variant_type", ""))
    gnomad_af: Optional[float] = _safe_float(row.get("gnomad_af"))
    gnomad_af_afr: Optional[float] = _safe_float(row.get("gnomad_af_afr"))
    clnsig = str(row.get("ClinVar Significance", ""))

    # ── PVS1: predicted loss-of-function ──────────────────────────────────────
    if any(term in ann for term in LOF_TERMS) or any(
        term in str(row.get("info_raw", "")).lower() for term in LOF_TERMS
    ):
        evidence_path.append("PVS1")

    # ── PS1: ClinVar pathogenic ───────────────────────────────────────────────
    if any(sig.lower() in clnsig.lower() for sig in PATHOGENIC_CLNSIG):
        evidence_path.append("PS1")

    # ── PM2: ultra-rare / absent in gnomAD ────────────────────────────────────
    pm2_result = classify_pm2_african(gnomad_af, gnomad_af_afr)
    if pm2_result["standard_call"] == "PM2_Supporting":
        evidence_path.append("PM2")
    if pm2_result["flag"] == "AFRICAN_CONTEXT_MISMATCH":
        african_context = pm2_result

    # ── PM4: in-frame INDEL ───────────────────────────────────────────────────
    if vtype == "INDEL":
        ref = str(row.get("ref", ""))
        alt = str(row.get("alt", ""))
        length_change = abs(len(ref) - len(alt))
        if 3 <= length_change <= 9 and length_change % 3 == 0:
            evidence_path.append("PM4")

    # ── BA1 / BS1: common allele ──────────────────────────────────────────────
    ba1_result = classify_ba1_african(gnomad_af, gnomad_af_afr)
    if gnomad_af is not None:
        if ba1_result["standard_call"] == "BA1_Strong":
            evidence_benign.append("BA1")
        elif gnomad_af >= AF_BS1:
            evidence_benign.append("BS1")

    if ba1_result["flag"] == "AFRICAN_CONTEXT_MISMATCH":
        african_context = ba1_result

    if any(sig.lower() in clnsig.lower() for sig in BENIGN_CLNSIG):
        evidence_benign.append("BS2")

    # ── Classification ────────────────────────────────────────────────────────
    classification = _classify(evidence_path, evidence_benign)

    result = {
        "acmg_class": classification,
        "acmg_path_evidence": ", ".join(evidence_path) or "—",
        "acmg_benign_evidence": ", ".join(evidence_benign) or "—",
    }

    if african_context:
        result["african_context_flag"] = african_context["flag"]
        result["african_context_recommendation"] = african_context["recommendation"]

    return result


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


def classify_pm2_african(overall_gnomad_af: Optional[float],
                        african_gnomad_af: Optional[float],
                        threshold: float = AFRICAN_PM2_THRESHOLD) -> dict:
    """African-contextualised PM2 criterion.

    Standard PM2: absent/rare in population databases (AF < 0.001)
    African PM2: checks gnomAD African subpopulation specifically.
    Flags when African AF significantly exceeds overall AF (population-specific variation).
    """
    result = {
        "criterion": "PM2",
        "standard_call": None,
        "african_call": None,
        "flag": None,
        "recommendation": None
    }

    # Standard PM2
    if overall_gnomad_af is None or overall_gnomad_af < threshold:
        result["standard_call"] = "PM2_Supporting"
    else:
        result["standard_call"] = "Not_PM2"

    # African contextualisation
    if african_gnomad_af is not None and overall_gnomad_af is not None:
        discrepancy_ratio = african_gnomad_af / (overall_gnomad_af + 1e-9)

        if discrepancy_ratio > AFRICAN_DISCREPANCY_RATIO and african_gnomad_af > AFRICAN_MIN_AF:
            result["african_call"] = "Not_PM2"
            result["flag"] = "AFRICAN_CONTEXT_MISMATCH"
            result["recommendation"] = (
                f"⚠️ African context detected: Overall gnomAD AF={overall_gnomad_af:.4f} suggests PM2, "
                f"but gnomAD African AF={african_gnomad_af:.4f} is substantially higher ({discrepancy_ratio:.1f}x). "
                f"This variant may represent benign African population-specific variation. "
                f"Recommend downgrading PM2 evidence for African ancestry patients."
            )
        else:
            result["african_call"] = result["standard_call"]
    elif african_gnomad_af is None:
        result["african_call"] = result["standard_call"]

    return result


def classify_ba1_african(overall_gnomad_af: Optional[float],
                        african_gnomad_af: Optional[float],
                        threshold: float = AFRICAN_BA1_THRESHOLD) -> dict:
    """African-contextualised BA1 criterion.

    Standard BA1: common allele (AF >= 5%, stand-alone benign)
    African BA1: checks if African AF contradicts benign call.
    Flags when African AF is rare but overall AF is high (may be enriched in other populations).
    """
    result = {
        "criterion": "BA1",
        "standard_call": None,
        "african_call": None,
        "flag": None,
        "recommendation": None
    }

    # Standard BA1
    if overall_gnomad_af is not None and overall_gnomad_af >= threshold:
        result["standard_call"] = "BA1_Strong"
    else:
        result["standard_call"] = "Not_BA1"

    # African contextualisation
    if african_gnomad_af is not None and overall_gnomad_af is not None:
        if overall_gnomad_af >= threshold and african_gnomad_af < (threshold * 0.2):
            result["african_call"] = "Not_BA1"
            result["flag"] = "AFRICAN_CONTEXT_MISMATCH"
            result["recommendation"] = (
                f"⚠️ African context detected: Overall gnomAD AF={overall_gnomad_af:.4f} suggests BA1, "
                f"but gnomAD African AF={african_gnomad_af:.4f} is much lower. "
                f"This variant may be enriched in other populations but rare in African cohorts. "
                f"Exercise caution calling BA1 for African ancestry patients without further evidence."
            )
        else:
            result["african_call"] = result["standard_call"]
    elif african_gnomad_af is None:
        result["african_call"] = result["standard_call"]

    return result


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add ACMG classification columns to every row of a DataFrame."""
    if df.empty:
        return df
    classifications = df.apply(classify_variant, axis=1, result_type="expand")
    return pd.concat([df.reset_index(drop=True), classifications.reset_index(drop=True)], axis=1)
