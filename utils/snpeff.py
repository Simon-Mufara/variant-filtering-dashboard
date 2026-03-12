"""SnpEff annotation parser — extracts ANN field from VCF INFO."""
import pandas as pd
import re

# Standard ANN field subfields (SnpEff v4+)
ANN_FIELDS = [
    "allele", "annotation", "annotation_impact", "gene_name",
    "gene_id", "feature_type", "feature_id", "transcript_biotype",
    "rank", "hgvs_c", "hgvs_p", "cdna_pos", "cds_pos",
    "protein_pos", "distance", "errors",
]

IMPACT_ORDER = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MODIFIER": 3}
IMPACT_COLORS = {
    "HIGH": "#d62728",
    "MODERATE": "#ff7f0e",
    "LOW": "#2ca02c",
    "MODIFIER": "#aec7e8",
}


def parse_snpeff(df: pd.DataFrame) -> pd.DataFrame:
    """Extract SnpEff ANN field into a flat DataFrame.

    Works on VCF DataFrames that have an 'info_raw' column OR
    DataFrames already flattened by load_vcf (reads from raw INFO string).
    Falls back gracefully if no ANN field present.
    """
    # If vcf_parser stored the raw INFO we can re-parse; otherwise skip
    info_col = None
    for col in ["info_raw", "INFO", "info"]:
        if col in df.columns:
            info_col = col
            break

    if info_col is None:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        ann_raw = _extract_ann(str(row.get(info_col, "")))
        if not ann_raw:
            continue
        for ann in ann_raw.split(","):
            parts = ann.split("|")
            rec = {ANN_FIELDS[i]: parts[i] if i < len(parts) else "" for i in range(len(ANN_FIELDS))}
            rec["chrom"] = row.get("chrom", "")
            rec["position"] = row.get("position", "")
            rec["ref"] = row.get("ref", "")
            rec["alt"] = row.get("alt", "")
            rec["quality"] = row.get("quality", None)
            records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()


def _extract_ann(info: str) -> str:
    """Extract the ANN= value from an INFO string."""
    match = re.search(r"(?:^|;)ANN=([^;]+)", info)
    return match.group(1) if match else ""


def impact_summary(ann_df: pd.DataFrame) -> pd.DataFrame:
    """Count variants per impact level."""
    if ann_df.empty or "annotation_impact" not in ann_df.columns:
        return pd.DataFrame(columns=["Impact", "Count"])
    counts = ann_df["annotation_impact"].value_counts().reset_index()
    counts.columns = ["Impact", "Count"]
    counts["order"] = counts["Impact"].map(IMPACT_ORDER).fillna(99)
    counts = counts.sort_values("order").drop(columns="order")
    return counts


def top_affected_genes(ann_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N most frequently affected genes."""
    if ann_df.empty or "gene_name" not in ann_df.columns:
        return pd.DataFrame(columns=["Gene", "Count", "High Impact"])
    g = ann_df[ann_df["gene_name"] != ""].groupby("gene_name").agg(
        Count=("gene_name", "count"),
        High_Impact=("annotation_impact", lambda x: (x == "HIGH").sum()),
    ).reset_index().rename(columns={"gene_name": "Gene", "High_Impact": "High Impact"})
    return g.sort_values("Count", ascending=False).head(n)
