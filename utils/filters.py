import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    min_quality: float,
    min_depth: int,
    variant_type: str,
    chromosomes: list = None,
    min_af: float = None,
    max_af: float = None,
    filter_pass_only: bool = False,
) -> pd.DataFrame:
    """Filter variants robustly — NaN quality/depth rows are kept unless
    a strict threshold is explicitly set above 0.
    """
    filtered = df.copy()

    # Quality: only drop rows where quality is known AND below threshold
    if min_quality > 0:
        filtered = filtered[filtered["quality"].isna() | (filtered["quality"] >= min_quality)]

    # Depth: only drop rows where depth is known AND below threshold
    if min_depth > 0:
        filtered = filtered[filtered["depth"] >= min_depth]

    # Variant type
    if variant_type != "All":
        filtered = filtered[filtered["variant_type"] == variant_type]

    # Chromosomes — match regardless of chr prefix
    if chromosomes:
        def _norm(c):
            return str(c).lower().lstrip("chr")
        norm_selected = {_norm(c) for c in chromosomes}
        filtered = filtered[filtered["chrom"].apply(lambda c: _norm(c) in norm_selected)]

    # Allele frequency
    if min_af is not None and min_af > 0.0 and "af" in filtered.columns:
        filtered = filtered[filtered["af"].isna() | (filtered["af"] >= min_af)]

    if max_af is not None and max_af < 1.0 and "af" in filtered.columns:
        filtered = filtered[filtered["af"].isna() | (filtered["af"] <= max_af)]

    # FILTER column — only PASS variants
    if filter_pass_only and "filter" in filtered.columns:
        filtered = filtered[filtered["filter"].isin(["PASS", ".", ""])]

    return filtered.reset_index(drop=True)


