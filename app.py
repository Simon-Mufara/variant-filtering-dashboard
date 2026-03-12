import os
import streamlit as st
import pandas as pd

from utils.vcf_parser import load_vcf
from utils.filters import apply_filters
from utils.plots import (
    chromosome_plot,
    variant_type_plot,
    quality_distribution,
    depth_distribution,
    af_scatter,
    tstv_plot,
    positional_track,
    annotate_with_genes,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Variant Filtering Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-container { background: #f0f2f6; border-radius: 8px; padding: 0.5rem; }
    section[data-testid="stSidebar"] { background: #1a1a2e; color: white; }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
st.sidebar.title("🧬 Variant Dashboard")
st.sidebar.markdown("---")

st.sidebar.header("📂 Data Input")
vcf_file = st.sidebar.file_uploader("Upload VCF file", type=["vcf"])
use_example = st.sidebar.checkbox("Use example VCF", value=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = None
if vcf_file:
    df_raw = load_vcf(vcf_file)
elif use_example:
    example_path = os.path.join(os.path.dirname(__file__), "data", "example.vcf")
    df_raw = load_vcf(example_path)

if df_raw is None:
    st.title("🧬 Genomic Variant Filtering Dashboard")
    st.info("Upload a VCF file or enable **Use example VCF** in the sidebar to get started.")
    st.stop()

# ── Detect sample columns ────────────────────────────────────────────────────
sample_cols = [c for c in df_raw.columns if c.startswith("sample_")]
samples = [c.replace("sample_", "").replace("_GT", "") for c in sample_cols]

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔧 Filters")

min_quality = st.sidebar.slider("Minimum Quality", 0, 100, 0)
min_depth = st.sidebar.slider("Minimum Depth", 0, 500, 0)

# Chromosome list — sorted in genomic order, dynamically from the VCF
def _chrom_sort_key(c):
    stripped = str(c).lower().lstrip("chr")
    return (0, int(stripped)) if stripped.isdigit() else (1, stripped)

all_chroms = sorted(df_raw["chrom"].unique().tolist(), key=_chrom_sort_key)
selected_chroms = st.sidebar.multiselect(
    f"Chromosomes ({len(all_chroms)} detected)",
    options=all_chroms,
    default=[],
    help="Leave empty to include all. Accepts any format: 1, chr1, chrX, MT, chrMT, etc.",
)

# Variant types — dynamically detected from the VCF
detected_types = sorted(df_raw["variant_type"].dropna().unique().tolist())
variant_type = st.sidebar.selectbox("Variant Type", ["All"] + detected_types)

filter_pass_only = st.sidebar.checkbox("PASS variants only", value=False)

if "af" in df_raw.columns and df_raw["af"].notna().any():
    af_range = st.sidebar.slider("Allele Frequency Range", 0.0, 1.0, (0.0, 1.0), step=0.01)
else:
    af_range = (None, None)

# ── Gene annotation toggle ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔬 Annotation")
annotate = st.sidebar.checkbox("Annotate with gene names (Ensembl)", value=False)
st.sidebar.caption("⚠️ Queries Ensembl REST API — slow for large datasets.")

# ── Apply filters ─────────────────────────────────────────────────────────────
df = apply_filters(
    df_raw,
    min_quality=min_quality,
    min_depth=min_depth,
    variant_type=variant_type,
    chromosomes=selected_chroms if selected_chroms else None,
    min_af=af_range[0],
    max_af=af_range[1],
    filter_pass_only=filter_pass_only,
)

# Optionally annotate
if annotate and not df.empty:
    with st.spinner("Querying Ensembl for gene names…"):
        df = annotate_with_genes(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 Genomic Variant Filtering Dashboard")
if samples:
    st.caption(f"Samples detected: **{', '.join(samples)}**")

# ── Summary metrics ───────────────────────────────────────────────────────────
pass_rate = round(len(df) / len(df_raw) * 100, 1) if len(df_raw) > 0 else 0
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Variants", len(df_raw))
c2.metric("Passing Variants", len(df))
c3.metric("Filtered Out", len(df_raw) - len(df))
c4.metric("Pass Rate", f"{pass_rate}%")
c5.metric("Chromosomes", df["chrom"].nunique())

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_distributions, tab_browser, tab_samples, tab_table = st.tabs([
    "📊 Overview",
    "📈 Distributions",
    "🗺️ Genome Browser",
    "👥 Multi-Sample",
    "📋 Data Table",
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    col1, col2 = st.columns(2)
    col1.plotly_chart(chromosome_plot(df), use_container_width=True)
    col2.plotly_chart(variant_type_plot(df), use_container_width=True)

    col3, col4 = st.columns(2)
    col3.plotly_chart(quality_distribution(df), use_container_width=True)
    col4.plotly_chart(tstv_plot(df), use_container_width=True)

# ── Tab 2: Distributions ──────────────────────────────────────────────────────
with tab_distributions:
    col1, col2 = st.columns(2)
    col1.plotly_chart(depth_distribution(df), use_container_width=True)
    if "af" in df.columns and df["af"].notna().any():
        col2.plotly_chart(af_scatter(df), use_container_width=True)
    else:
        col2.info("AF data not available in this VCF.")

# ── Tab 3: Genome Browser ─────────────────────────────────────────────────────
with tab_browser:
    st.subheader("🗺️ Positional Variant Track")
    chrom_select = st.selectbox(
        "Select chromosome to view", options=sorted(df["chrom"].unique().tolist())
    )
    st.plotly_chart(positional_track(df, chrom_select), use_container_width=True)

# ── Tab 4: Multi-sample ───────────────────────────────────────────────────────
with tab_samples:
    st.subheader("👥 Per-Sample Genotypes")
    if sample_cols:
        display_cols = ["chrom", "position", "ref", "alt", "variant_type", "quality", "depth"] + sample_cols
        st.dataframe(df[display_cols], use_container_width=True, height=400)

        st.markdown("**Genotype Counts per Sample**")
        for col in sample_cols:
            sample_name = col.replace("sample_", "").replace("_GT", "")
            counts = df[col].value_counts().reset_index()
            counts.columns = ["Genotype", "Count"]
            st.markdown(f"*{sample_name}*")
            st.dataframe(counts, use_container_width=False)
    else:
        st.info("No sample genotype data found in this VCF.")

# ── Tab 5: Data Table ─────────────────────────────────────────────────────────
with tab_table:
    st.subheader("📋 Filtered Variants")
    st.dataframe(df, use_container_width=True, height=420)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Variants (CSV)",
        data=csv,
        file_name="filtered_variants.csv",
        mime="text/csv",
    )

