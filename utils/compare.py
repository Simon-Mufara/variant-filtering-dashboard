"""VCF comparison utilities — compare two loaded DataFrames."""
import pandas as pd


def _variant_key(df: pd.DataFrame) -> pd.Series:
    """Create a unique key per variant: chrom:pos:ref:alt."""
    return (
        df["chrom"].astype(str) + ":" +
        df["position"].astype(str) + ":" +
        df["ref"].astype(str) + ":" +
        df["alt"].astype(str)
    )


def compare_vcfs(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Compare two VCF DataFrames.

    Returns a dict with:
        shared      — variants present in both
        only_a      — variants only in A
        only_b      — variants only in B
        concordance — % shared / total unique
        summary     — text summary dict
    """
    keys_a = set(_variant_key(df_a))
    keys_b = set(_variant_key(df_b))

    shared_keys = keys_a & keys_b
    only_a_keys = keys_a - keys_b
    only_b_keys = keys_b - keys_a
    total_unique = len(keys_a | keys_b)

    key_col_a = _variant_key(df_a)
    key_col_b = _variant_key(df_b)

    shared_a = df_a[key_col_a.isin(shared_keys)].copy()
    only_a   = df_a[key_col_a.isin(only_a_keys)].copy()
    only_b   = df_b[key_col_b.isin(only_b_keys)].copy()

    concordance = round(len(shared_keys) / total_unique * 100, 2) if total_unique > 0 else 0.0

    return {
        "shared": shared_a,
        "only_a": only_a,
        "only_b": only_b,
        "n_shared": len(shared_keys),
        "n_only_a": len(only_a_keys),
        "n_only_b": len(only_b_keys),
        "concordance": concordance,
        "total_unique": total_unique,
    }


def concordance_by_type(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Concordance breakdown per variant type."""
    rows = []
    for vtype in sorted(set(df_a["variant_type"].dropna()) | set(df_b["variant_type"].dropna())):
        a = df_a[df_a["variant_type"] == vtype]
        b = df_b[df_b["variant_type"] == vtype]
        ka = set(_variant_key(a))
        kb = set(_variant_key(b))
        sh = len(ka & kb)
        total = len(ka | kb)
        rows.append({
            "Variant Type": vtype,
            "In A": len(ka),
            "In B": len(kb),
            "Shared": sh,
            "Concordance (%)": round(sh / total * 100, 1) if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)
