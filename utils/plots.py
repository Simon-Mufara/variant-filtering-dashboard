import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Consistent colour map
TYPE_COLORS = {"SNP": "#1f77b4", "INDEL": "#ff7f0e"}
TS_TV_COLORS = {"Ts": "#2ca02c", "Tv": "#d62728"}


def chromosome_plot(df: pd.DataFrame) -> go.Figure:
    """Bar chart of variant counts per chromosome."""
    counts = df["chrom"].value_counts().reset_index()
    counts.columns = ["chromosome", "count"]
    fig = px.bar(
        counts, x="chromosome", y="count",
        title="Variants per Chromosome",
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(xaxis_title="Chromosome", yaxis_title="Count")
    return fig


def variant_type_plot(df: pd.DataFrame) -> go.Figure:
    """Pie chart of variant type distribution."""
    counts = df["variant_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.pie(
        counts, names="type", values="count",
        title="Variant Type Distribution",
        color="type", color_discrete_map=TYPE_COLORS,
    )
    return fig


def quality_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of quality score distribution."""
    fig = px.histogram(
        df, x="quality", nbins=50,
        title="Quality Score Distribution",
        color_discrete_sequence=["#9467bd"],
    )
    fig.update_layout(xaxis_title="Quality Score", yaxis_title="Count")
    return fig


# --- Feature 1: Additional visualizations ---

def depth_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of read depth distribution."""
    fig = px.histogram(
        df, x="depth", nbins=50,
        title="Read Depth (DP) Distribution",
        color_discrete_sequence=["#ff7f0e"],
    )
    fig.update_layout(xaxis_title="Read Depth", yaxis_title="Count")
    return fig


def af_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Allele Frequency vs Quality Score."""
    plot_df = df.dropna(subset=["af"])
    fig = px.scatter(
        plot_df, x="af", y="quality",
        color="variant_type",
        color_discrete_map=TYPE_COLORS,
        title="Allele Frequency vs Quality Score",
        labels={"af": "Allele Frequency", "quality": "Quality Score"},
        opacity=0.7,
    )
    return fig


def tstv_plot(df: pd.DataFrame) -> go.Figure:
    """Bar chart of Transition/Transversion ratio for SNPs."""
    from utils.vcf_parser import classify_ts_tv

    snps = df[df["variant_type"] == "SNP"].copy()
    if snps.empty:
        fig = go.Figure()
        fig.update_layout(title="Ts/Tv Ratio (no SNPs in selection)")
        return fig

    snps["tstv"] = snps.apply(lambda r: classify_ts_tv(r["ref"], r["alt"]), axis=1)
    counts = snps["tstv"].value_counts().reset_index()
    counts.columns = ["type", "count"]

    ts = counts.loc[counts["type"] == "Ts", "count"].sum()
    tv = counts.loc[counts["type"] == "Tv", "count"].sum()
    ratio = round(ts / tv, 3) if tv > 0 else float("inf")

    fig = px.bar(
        counts, x="type", y="count",
        color="type", color_discrete_map=TS_TV_COLORS,
        title=f"Transition / Transversion  (Ts/Tv = {ratio})",
        labels={"type": "Class", "count": "Count"},
    )
    fig.update_layout(showlegend=False)
    return fig


# --- Feature 2: Genome browser / positional track ---

def positional_track(df: pd.DataFrame, chrom: str) -> go.Figure:
    """Scatter plot of variant positions along a chromosome."""
    sub = df[df["chrom"] == chrom].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No variants on {chrom}")
        return fig

    fig = px.scatter(
        sub, x="position", y="quality",
        color="variant_type",
        color_discrete_map=TYPE_COLORS,
        hover_data=["ref", "alt", "depth", "af"],
        title=f"Variant Positions — {chrom}",
        labels={"position": "Genomic Position", "quality": "Quality Score"},
        opacity=0.75,
    )
    fig.update_layout(xaxis_title=f"{chrom} position", yaxis_title="Quality")
    return fig


# --- Feature 3: Gene annotation table ---

def annotate_with_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Query Ensembl REST API to add gene names for each variant."""
    import requests

    BASE = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    gene_names = []
    for _, row in df.iterrows():
        chrom = str(row["chrom"]).replace("chr", "")
        pos = int(row["position"])
        try:
            url = f"{BASE}/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene"
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.ok:
                genes = resp.json()
                names = ", ".join(
                    g.get("external_name", g.get("gene_id", "")) for g in genes if g.get("feature_type") == "gene"
                )
                gene_names.append(names if names else "—")
            else:
                gene_names.append("—")
        except Exception:
            gene_names.append("—")

    df = df.copy()
    df.insert(4, "gene", gene_names)
    return df

