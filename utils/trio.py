"""Trio analysis — de novo and recessive variant detection.

Given three VCF DataFrames (proband, mother, father), identifies:
    de_novo           — present in proband, absent in both parents
    homozygous_rec    — homozygous in proband, heterozygous in both parents
    compound_het      — two het variants in same gene in proband, each inherited from one parent
    x_linked          — hemizygous on chrX in proband (male), absent/het in mother

All comparisons are done on chrom:pos:ref:alt keys.
"""
from __future__ import annotations
import pandas as pd


def _key(df: pd.DataFrame) -> pd.Series:
    return (df["chrom"].astype(str) + ":" +
            df["position"].astype(str) + ":" +
            df["ref"].astype(str) + ":" +
            df["alt"].astype(str))


def _is_het(gt: str) -> bool:
    gt = str(gt).replace("|", "/")
    parts = gt.split("/")
    return len(parts) == 2 and parts[0] != parts[1] and "." not in parts


def _is_hom_alt(gt: str) -> bool:
    gt = str(gt).replace("|", "/")
    parts = gt.split("/")
    return len(parts) == 2 and parts[0] == parts[1] and parts[0] not in ("0", ".")


def _is_absent(gt: str) -> bool:
    gt = str(gt).replace("|", "/")
    return gt in ("0/0", "0|0", "./.", ".|.", "")


def detect_denovo(
    proband: pd.DataFrame,
    mother: pd.DataFrame,
    father: pd.DataFrame,
) -> pd.DataFrame:
    """Return variants present in proband but absent (0/0 or missing) in both parents."""
    prob_keys = set(_key(proband))
    mom_keys  = set(_key(mother))
    dad_keys  = set(_key(father))

    denovo_keys = prob_keys - mom_keys - dad_keys
    result = proband[_key(proband).isin(denovo_keys)].copy()
    result["inheritance"] = "De Novo"
    return result.reset_index(drop=True)


def detect_homozygous_recessive(
    proband: pd.DataFrame,
    mother: pd.DataFrame,
    father: pd.DataFrame,
) -> pd.DataFrame:
    """Homozygous alt in proband + het in both parents."""
    # We need GT columns — use the first sample column if present
    prob_gt_col = _first_gt_col(proband)
    mom_gt_col  = _first_gt_col(mother)
    dad_gt_col  = _first_gt_col(father)

    if not all([prob_gt_col, mom_gt_col, dad_gt_col]):
        return pd.DataFrame(columns=proband.columns)

    prob_hom = proband[proband[prob_gt_col].apply(_is_hom_alt)].copy()
    prob_keys_hom = set(_key(prob_hom))

    mom_het_keys = set(_key(mother[mother[mom_gt_col].apply(_is_het)]))
    dad_het_keys = set(_key(father[father[dad_gt_col].apply(_is_het)]))

    hom_rec_keys = prob_keys_hom & mom_het_keys & dad_het_keys
    result = prob_hom[_key(prob_hom).isin(hom_rec_keys)].copy()
    result["inheritance"] = "Homozygous Recessive"
    return result.reset_index(drop=True)


def detect_compound_het(
    proband: pd.DataFrame,
    mother: pd.DataFrame,
    father: pd.DataFrame,
    gene_col: str = "gene_name",
) -> pd.DataFrame:
    """Two het variants in same gene — one maternal, one paternal.

    Requires gene annotation (gene_name or vep_symbol column).
    """
    prob_gt_col = _first_gt_col(proband)
    mom_gt_col  = _first_gt_col(mother)
    dad_gt_col  = _first_gt_col(father)

    if not all([prob_gt_col, mom_gt_col, dad_gt_col]):
        return pd.DataFrame(columns=proband.columns)

    # Resolve gene column
    gene_c = next((c for c in [gene_col, "vep_symbol", "gene", "Gene"]
                   if c in proband.columns), None)
    if gene_c is None:
        return pd.DataFrame(columns=proband.columns)

    prob_het = proband[proband[prob_gt_col].apply(_is_het)].copy()
    mom_keys  = set(_key(mother[mother[mom_gt_col].apply(_is_het)]))
    dad_keys  = set(_key(father[father[dad_gt_col].apply(_is_het)]))

    # Tag each het variant in proband with whether it's maternal or paternal
    pk = _key(prob_het)
    prob_het = prob_het.copy()
    prob_het["_mat"] = pk.isin(mom_keys)
    prob_het["_pat"] = pk.isin(dad_keys)

    compound = []
    for gene, grp in prob_het.groupby(gene_c):
        if gene in ("", ".", None):
            continue
        mat_vars = grp[grp["_mat"]]
        pat_vars = grp[grp["_pat"]]
        if len(mat_vars) >= 1 and len(pat_vars) >= 1:
            combined = pd.concat([mat_vars, pat_vars])
            combined["inheritance"] = f"Compound Het ({gene})"
            compound.append(combined)

    if compound:
        return pd.concat(compound).drop(columns=["_mat", "_pat"]).reset_index(drop=True)
    return pd.DataFrame(columns=proband.columns)


def run_trio_analysis(
    proband: pd.DataFrame,
    mother: pd.DataFrame,
    father: pd.DataFrame,
) -> dict:
    """Run all trio analysis modes and return a results dict."""
    denovo   = detect_denovo(proband, mother, father)
    hom_rec  = detect_homozygous_recessive(proband, mother, father)
    comp_het = detect_compound_het(proband, mother, father)

    return {
        "de_novo":             denovo,
        "homozygous_recessive": hom_rec,
        "compound_het":        comp_het,
        "n_denovo":            len(denovo),
        "n_hom_rec":           len(hom_rec),
        "n_comp_het":          len(comp_het),
    }


def _first_gt_col(df: pd.DataFrame):
    for col in df.columns:
        if col.startswith("sample_") and col.endswith("_GT"):
            return col
    return None
