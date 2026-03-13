"""gnomAD population frequency lookup via the public GraphQL API (v2/v3)."""
from __future__ import annotations
import requests
import pandas as pd
from utils.logger import log

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
_TIMEOUT = 10  # seconds


def lookup_gnomad(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    dataset: str = "gnomad_r4",
) -> dict | None:
    """Query gnomAD for population allele frequencies for a single variant.

    Args:
        chrom:   chromosome (with or without 'chr' prefix)
        pos:     1-based position
        ref:     reference allele
        alt:     alternate allele
        dataset: 'gnomad_r4' (GRCh38) or 'gnomad_r2_1' (GRCh37)

    Returns:
        dict with keys: af, af_afr, af_eas, af_nfe, af_sas, af_amr, af_fin,
                        homozygote_count, coverage  — or None if not found.
    """
    chrom_clean = chrom.lstrip("chr") if chrom.lower().startswith("chr") else chrom
    variant_id = f"{chrom_clean}-{pos}-{ref}-{alt}"

    query = """
    query VariantDetails($variantId: String!, $dataset: DatasetId!) {
      variant(variantId: $variantId, dataset: $dataset) {
        genome {
          af
          populations { id af }
          homozygote_count
        }
        coverage {
          genome { median }
        }
      }
    }
    """
    try:
        resp = requests.post(
            GNOMAD_API,
            json={"query": query, "variables": {"variantId": variant_id, "dataset": dataset}},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("variant") or {}
        genome = data.get("genome") or {}
        if not genome:
            return None

        pops = {p["id"]: p["af"] for p in (genome.get("populations") or [])}
        return {
            "gnomad_af": genome.get("af"),
            "gnomad_af_afr": pops.get("afr"),
            "gnomad_af_eas": pops.get("eas"),
            "gnomad_af_nfe": pops.get("nfe"),
            "gnomad_af_sas": pops.get("sas"),
            "gnomad_af_amr": pops.get("amr"),
            "gnomad_af_fin": pops.get("fin"),
            "gnomad_homozygotes": genome.get("homozygote_count"),
            "gnomad_coverage": (data.get("coverage") or {}).get("genome", {}).get("median"),
        }
    except Exception as exc:
        log.warning("gnomAD lookup failed for %s: %s", variant_id, exc)
        return None


def annotate_gnomad(df: pd.DataFrame, dataset: str = "gnomad_r4", max_variants: int = 50,
                    genome_build: str = "GRCh38") -> pd.DataFrame:
    """Add gnomAD columns to a DataFrame.

    Only annotates up to *max_variants* rows to avoid rate-limiting.
    """
    if df.empty:
        return df

    # Select dataset based on genome build
    if "GRCh37" in genome_build:
        dataset = "gnomad_r2_1"

    results = []
    for _, row in df.head(max_variants).iterrows():
        result = lookup_gnomad(
            str(row.get("chrom", "")),
            int(row.get("position", 0)),
            str(row.get("ref", "")),
            str(row.get("alt", "")),
            dataset=dataset,
        )
        results.append(result or {})

    # Pad remaining rows with empty dicts
    results += [{}] * max(0, len(df) - max_variants)

    gnomad_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df.reset_index(drop=True), gnomad_df.reset_index(drop=True)], axis=1)
