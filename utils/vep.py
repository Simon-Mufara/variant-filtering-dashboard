"""Ensembl VEP REST API — full consequence annotation per variant.

Docs: https://rest.ensembl.org/documentation/info/vep_hgvs_post
Rate limit: 200 variants per POST, ~15 req/sec.

Returns per-variant:
    consequence   — most severe consequence term
    impact        — HIGH / MODERATE / LOW / MODIFIER
    gene_id       — Ensembl gene ID
    gene_symbol   — HGNC symbol
    transcript_id — canonical transcript
    hgvsc         — HGVS coding notation
    hgvsp         — HGVS protein notation
    sift_pred     — tolerated / deleterious
    polyphen_pred — benign / possibly_damaging / probably_damaging
    cadd_phred    — if CADD plugin enabled on Ensembl
    existing_var  — known rsID / ClinVar IDs
"""
from __future__ import annotations
import time
import requests
import pandas as pd
from utils.logger import log

VEP_URL = "https://rest.ensembl.org/vep/human/region"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
BATCH_SIZE = 100
RATE_DELAY = 0.1   # seconds between batches (be a good citizen)


def annotate_vep(df: pd.DataFrame, max_variants: int = 100) -> pd.DataFrame:
    """Run Ensembl VEP on a VCF DataFrame (up to max_variants rows).

    Adds columns: vep_consequence, vep_impact, vep_gene, vep_symbol,
                  vep_transcript, vep_hgvsc, vep_hgvsp, vep_sift, vep_polyphen,
                  vep_cadd_phred, vep_existing
    """
    if df.empty:
        return df

    subset = df.head(max_variants).copy()
    variants = _build_vep_input(subset)

    results = {}
    for i in range(0, len(variants), BATCH_SIZE):
        batch = variants[i: i + BATCH_SIZE]
        batch_results = _query_vep(batch)
        results.update(batch_results)
        time.sleep(RATE_DELAY)

    ann_rows = [_extract_top_annotation(results.get(v, {})) for v in variants]
    # Pad remaining rows
    ann_rows += [_empty_annotation()] * max(0, len(df) - max_variants)

    ann_df = pd.DataFrame(ann_rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), ann_df.reset_index(drop=True)], axis=1)


def _build_vep_input(df: pd.DataFrame) -> list[str]:
    """Convert DataFrame rows to VEP region strings: CHROM POS . REF ALT . . ."""
    variants = []
    for _, row in df.iterrows():
        chrom = str(row.get("chrom", "")).lstrip("chr")
        pos = int(row.get("position", 0))
        ref = str(row.get("ref", "N"))
        alt = str(row.get("alt", "N"))
        # VEP region format: chrom start end allele strand
        # For SNPs/INDELs, use: chrom:start-end ref/alt
        end = pos + max(len(ref), 1) - 1
        variants.append(f"{chrom} {pos} {end} {ref}/{alt} 1")
    return variants


def _query_vep(variants: list[str]) -> dict:
    """POST to VEP API and return dict keyed by variant string."""
    try:
        resp = requests.post(
            VEP_URL,
            headers=HEADERS,
            json={"variants": variants},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        out = {}
        for item in data:
            key = item.get("input", "")
            out[key] = item
        return out
    except Exception as exc:
        log.warning("VEP API error: %s", exc)
        return {}


def _extract_top_annotation(vep_item: dict) -> dict:
    """Pull the most severe consequence from a VEP response item."""
    if not vep_item:
        return _empty_annotation()

    consequences = vep_item.get("transcript_consequences", [])
    if not consequences:
        consequences = vep_item.get("intergenic_consequences", [])

    # Sort by severity — VEP returns them ranked but let's be explicit
    top = consequences[0] if consequences else {}

    # SIFT / PolyPhen
    sift = top.get("sift_prediction", "")
    if top.get("sift_score") is not None:
        sift = f"{sift} ({top['sift_score']:.3f})"
    poly = top.get("polyphen_prediction", "")
    if top.get("polyphen_score") is not None:
        poly = f"{poly} ({top['polyphen_score']:.3f})"

    # Existing variants
    existing = ", ".join(
        v.get("id", "") for v in vep_item.get("colocated_variants", [])
        if v.get("id", "").startswith("rs") or "ClinVar" in str(v)
    ) or "—"

    return {
        "vep_consequence": _join_terms(top.get("consequence_terms", [])),
        "vep_impact": top.get("impact", ""),
        "vep_gene": top.get("gene_id", ""),
        "vep_symbol": top.get("gene_symbol", ""),
        "vep_transcript": top.get("transcript_id", ""),
        "vep_hgvsc": top.get("hgvsc", ""),
        "vep_hgvsp": top.get("hgvsp", ""),
        "vep_sift": sift,
        "vep_polyphen": poly,
        "vep_cadd_phred": top.get("cadd_phred", ""),
        "vep_existing": existing,
    }


def _empty_annotation() -> dict:
    return {k: "" for k in [
        "vep_consequence", "vep_impact", "vep_gene", "vep_symbol",
        "vep_transcript", "vep_hgvsc", "vep_hgvsp", "vep_sift",
        "vep_polyphen", "vep_cadd_phred", "vep_existing",
    ]}


def _join_terms(terms: list) -> str:
    return ", ".join(t.replace("_", " ") for t in terms)
