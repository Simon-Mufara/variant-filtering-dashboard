"""Variant statistics — summary metrics beyond basic counts."""
import pandas as pd
import numpy as np
from utils.vcf_parser import classify_ts_tv


def variant_stats(df: pd.DataFrame) -> dict:
    """Compute comprehensive variant statistics."""
    stats = {}
    stats["total"] = len(df)

    if "variant_type" in df.columns:
        type_counts = df["variant_type"].value_counts().to_dict()
        stats["snp_count"] = type_counts.get("SNP", 0)
        stats["indel_count"] = type_counts.get("INDEL", 0)
        stats["mnp_count"] = type_counts.get("MNP", 0)
        stats["sv_count"] = type_counts.get("SV", 0)

    # Ts/Tv ratio
    snps = df[df.get("variant_type", pd.Series()) == "SNP"] if "variant_type" in df.columns else pd.DataFrame()
    if not snps.empty:
        tstv = snps.apply(lambda r: classify_ts_tv(r["ref"], r["alt"]), axis=1)
        ts = (tstv == "Ts").sum()
        tv = (tstv == "Tv").sum()
        stats["ts_count"] = int(ts)
        stats["tv_count"] = int(tv)
        stats["tstv_ratio"] = round(ts / tv, 3) if tv > 0 else float("inf")
    else:
        stats["ts_count"] = stats["tv_count"] = 0
        stats["tstv_ratio"] = None

    # Genotype stats from sample columns
    sample_cols = [c for c in df.columns if c.startswith("sample_") and c.endswith("_GT")]
    het_counts, hom_alt_counts, hom_ref_counts, missing_counts = [], [], [], []
    for col in sample_cols:
        gts = df[col].astype(str)
        het_counts.append((gts.str.contains(r"0[/|]1|1[/|]0", regex=True)).sum())
        hom_alt_counts.append((gts.str.match(r"^1[/|]1$")).sum())
        hom_ref_counts.append((gts.str.match(r"^0[/|]0$")).sum())
        missing_counts.append((gts.str.contains(r"\.", regex=True)).sum())

    if sample_cols:
        stats["het"] = int(np.mean(het_counts))
        stats["hom_alt"] = int(np.mean(hom_alt_counts))
        stats["hom_ref"] = int(np.mean(hom_ref_counts))
        stats["missing"] = int(np.mean(missing_counts))
        stats["het_hom_ratio"] = round(stats["het"] / stats["hom_alt"], 3) if stats["hom_alt"] > 0 else None
        stats["missingness_pct"] = round(stats["missing"] / len(df) * 100, 2) if len(df) > 0 else 0
    else:
        stats.update({"het": None, "hom_alt": None, "hom_ref": None,
                      "missing": None, "het_hom_ratio": None, "missingness_pct": None})

    # Quality stats
    if "quality" in df.columns:
        q = df["quality"].dropna()
        stats["mean_qual"] = round(q.mean(), 1) if len(q) > 0 else None
        stats["median_qual"] = round(q.median(), 1) if len(q) > 0 else None

    # Depth stats
    if "depth" in df.columns:
        d = df["depth"].dropna()
        stats["mean_depth"] = round(d.mean(), 1) if len(d) > 0 else None
        stats["median_depth"] = round(d.median(), 1) if len(d) > 0 else None

    return stats


def depth_per_chrom(df: pd.DataFrame) -> pd.DataFrame:
    """Mean read depth per chromosome."""
    if "chrom" not in df.columns or "depth" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("chrom")["depth"]
        .agg(mean_depth="mean", median_depth="median", count="count")
        .round(1)
        .reset_index()
        .rename(columns={"chrom": "Chromosome", "mean_depth": "Mean Depth",
                         "median_depth": "Median Depth", "count": "Variant Count"})
    )


def clinvar_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ClinVar CLNSIG field from variants if present."""
    import re
    if "info_raw" not in df.columns and "INFO" not in df.columns:
        return pd.DataFrame()

    info_col = "info_raw" if "info_raw" in df.columns else "INFO"
    records = []
    for _, row in df.iterrows():
        info = str(row.get(info_col, ""))
        m = re.search(r"CLNSIG=([^;]+)", info)
        clnsig = m.group(1) if m else "Unknown"
        m2 = re.search(r"CLNDN=([^;]+)", info)
        clndn = m2.group(1).replace("_", " ") if m2 else "—"
        records.append({
            "chrom": row.get("chrom", ""),
            "position": row.get("position", ""),
            "ref": row.get("ref", ""),
            "alt": row.get("alt", ""),
            "ClinVar Significance": clnsig,
            "Disease": clndn,
        })
    return pd.DataFrame(records)


def allele_balance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute allele balance (AB) per variant if AD/AO columns present."""
    import re
    if "info_raw" not in df.columns:
        return pd.DataFrame()
    records = []
    for _, row in df.iterrows():
        info = str(row.get("info_raw", ""))
        m = re.search(r"AD=(\d+),(\d+)", info)
        if m:
            ref_d, alt_d = int(m.group(1)), int(m.group(2))
            total = ref_d + alt_d
            ab = round(alt_d / total, 3) if total > 0 else None
            records.append({"chrom": row.get("chrom", ""), "position": row.get("position", 0),
                             "ref_depth": ref_d, "alt_depth": alt_d, "allele_balance": ab})
    return pd.DataFrame(records) if records else pd.DataFrame()


def variant_density(df: pd.DataFrame, bin_size_mb: int = 10) -> pd.DataFrame:
    """Count variants per genomic bin (Mb) per chromosome."""
    if "chrom" not in df.columns or "position" not in df.columns:
        return pd.DataFrame()
    d = df[["chrom", "position"]].copy()
    d["bin"] = (d["position"] // (bin_size_mb * 1_000_000)) * bin_size_mb
    return d.groupby(["chrom", "bin"]).size().reset_index(name="count")


def missingness_per_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sample missingness rate."""
    sample_cols = [c for c in df.columns if c.startswith("sample_") and c.endswith("_GT")]
    if not sample_cols:
        return pd.DataFrame()
    records = []
    for col in sample_cols:
        sample = col.replace("sample_", "").replace("_GT", "")
        gts = df[col].astype(str)
        missing = gts.str.contains(r"\.", regex=True).sum()
        records.append({"sample": sample, "total": len(df),
                        "missing": int(missing),
                        "missingness_pct": round(missing / max(len(df), 1) * 100, 2)})
    return pd.DataFrame(records)
