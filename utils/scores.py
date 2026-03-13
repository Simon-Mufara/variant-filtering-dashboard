"""Pathogenicity predictor score parsing — CADD, SpliceAI, REVEL from INFO field.

These scores are written into VCF INFO by annotation tools:
    CADD:    CADD_PHRED=xx (from CADD plugin / bcftools annotate)
    SpliceAI: SpliceAI_pred_DS_AG=x|x|x|x (SpliceAI VCF)
    REVEL:   REVEL=0.xxx (dbnsfp annotation)
    AlphaMissense: AM_PATHOGENICITY=x.xxx;AM_CLASS=likely_pathogenic
"""
from __future__ import annotations
import re
import pandas as pd


_CADD_RE      = re.compile(r"CADD(?:_PHRED)?=([0-9.]+)")
_REVEL_RE     = re.compile(r"REVEL=([0-9.]+)")
_AM_SCORE_RE  = re.compile(r"AM_PATHOGENICITY=([0-9.]+)")
_AM_CLASS_RE  = re.compile(r"AM_CLASS=([^;]+)")
# SpliceAI: take max of the four delta scores
_SPLICEAI_RE  = re.compile(r"SpliceAI_pred_DS_(?:AG|AL|DG|DL)=([0-9.]+)")


def parse_predictor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Extract CADD, SpliceAI, REVEL, AlphaMissense from info_raw column.

    Adds columns (all float, NaN if absent):
        cadd_phred, revel_score, spliceai_max_delta,
        alphamissense_score, alphamissense_class
    """
    if "info_raw" not in df.columns or df.empty:
        return df

    scores = df["info_raw"].apply(_parse_row)
    score_df = pd.DataFrame(scores.tolist(), index=df.index)
    return pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)


def _parse_row(info: str) -> dict:
    info = str(info)

    cadd = _first_float(_CADD_RE, info)
    revel = _first_float(_REVEL_RE, info)
    am_score = _first_float(_AM_SCORE_RE, info)
    am_class = _first_str(_AM_CLASS_RE, info)

    # SpliceAI max delta score
    splice_scores = [float(m.group(1)) for m in _SPLICEAI_RE.finditer(info)]
    splice_max = max(splice_scores) if splice_scores else None

    return {
        "cadd_phred": cadd,
        "revel_score": revel,
        "spliceai_max_delta": splice_max,
        "alphamissense_score": am_score,
        "alphamissense_class": am_class or "",
    }


def _first_float(pattern, text) -> float | None:
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _first_str(pattern, text) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def score_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for each predictor score column."""
    rows = []
    for col, label, threshold, danger in [
        ("cadd_phred",          "CADD Phred",            20,  "≥20 = likely deleterious"),
        ("revel_score",         "REVEL",                  0.5, "≥0.5 = likely pathogenic"),
        ("spliceai_max_delta",  "SpliceAI Max Delta",     0.5, "≥0.5 = likely splice disruption"),
        ("alphamissense_score", "AlphaMissense",          0.5, "≥0.5 = likely pathogenic"),
    ]:
        if col in df.columns:
            present = df[col].dropna()
            if len(present) > 0:
                rows.append({
                    "Predictor": label,
                    "Variants with score": len(present),
                    "Mean": round(present.mean(), 3),
                    "Median": round(present.median(), 3),
                    f"Above threshold ({threshold})": int((present >= threshold).sum()),
                    "Threshold note": danger,
                })
    return pd.DataFrame(rows)
