import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    min_quality: float,
    min_depth: int,
    variant_type: str,
    chromosomes: list = None,
    min_af: float = None,
    max_af: float = None,
) -> pd.DataFrame:
    """Filter variants by quality, depth, type, chromosome, and allele frequency."""
    filtered = df[
        (df["quality"] >= min_quality) &
        (df["depth"] >= min_depth)
    ]

    if variant_type != "All":
        filtered = filtered[filtered["variant_type"] == variant_type]

    if chromosomes:
        filtered = filtered[filtered["chrom"].isin(chromosomes)]

    if min_af is not None and "af" in filtered.columns:
        filtered = filtered[filtered["af"].isna() | (filtered["af"] >= min_af)]

    if max_af is not None and "af" in filtered.columns:
        filtered = filtered[filtered["af"].isna() | (filtered["af"] <= max_af)]

    return filtered.reset_index(drop=True)

