"""Variant Analysis Suite — main Streamlit application."""
import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px

from config import DEFAULT_MIN_QUAL, DEFAULT_MIN_DP
from utils.auth import require_auth
from utils.vcf_parser import load_vcf
from utils.validator import validate_vcf
from utils.filters import apply_filters
from utils.compare import compare_vcfs, concordance_by_type
from utils.snpeff import parse_snpeff, impact_summary, top_affected_genes, IMPACT_COLORS
from utils.stats import variant_stats, depth_per_chrom, clinvar_significance
from utils.acmg import classify_dataframe
from utils.gnomad import annotate_gnomad
from utils.prioritize import prioritize_dataframe
from utils.gene_panel import list_panels, get_panel_genes, parse_custom_panel, filter_to_panel
from utils.vep import annotate_vep
from utils.trio import run_trio_analysis
from utils.scores import parse_predictor_scores, score_summary
from utils.report import generate_report
from utils.pdf_report import generate_pdf, pdf_available
from utils.logger import log
from utils.plots import (
    chromosome_plot, variant_type_plot, quality_distribution,
    depth_distribution, af_scatter, tstv_plot, positional_track, annotate_with_genes,
)

# ── Auth gate (password via st.secrets["APP_PASSWORD"]) ──────────────────────
require_auth()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Variant Analysis Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Simon-Mufara/variant-filtering-dashboard",
        "Report a bug": "https://github.com/Simon-Mufara/variant-filtering-dashboard/issues",
        "About": "# Variant Analysis Suite\nIndustry-grade genomic variant analysis platform.",
    },
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 0.8rem; padding-bottom: 1rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%); }
  section[data-testid="stSidebar"] .stMarkdown,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span { color: #cbd5e1 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #93c5fd !important; }
  section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.8rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  [data-testid="metric-container"] label { color: #64748b; font-size: .8rem; font-weight: 600; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1e293b; font-weight: 700; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f1f5f9;
    border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 6px; padding: .4rem .9rem;
    font-size: .85rem; font-weight: 500; color: #64748b; }
  .stTabs [aria-selected="true"] { background: white !important; color: #1e40af !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.1); font-weight: 600; }

  /* Dividers */
  hr { border-color: #e2e8f0; }

  /* Info boxes */
  .stAlert { border-radius: 8px; }

  /* Priority tiers */
  .tier-high   { color: #dc2626; font-weight: 700; }
  .tier-medium { color: #ea580c; font-weight: 700; }
  .tier-low    { color: #16a34a; font-weight: 700; }

  /* Section headers */
  .section-header { font-size: 1.1rem; font-weight: 700; color: #1e293b;
                    border-left: 4px solid #3b82f6; padding-left: .6rem;
                    margin: 1rem 0 .5rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Parsing VCF…")
def _cached_load_path(path: str) -> pd.DataFrame:
    log.info("Loading VCF from path: %s", path)
    return load_vcf(path)


@st.cache_data(show_spinner="Parsing VCF…")
def _cached_load_bytes(data: bytes, name: str) -> pd.DataFrame:
    log.info("Loading uploaded VCF: %s (%d bytes)", name, len(data))
    return load_vcf(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def _cached_annotate_vep(df: pd.DataFrame) -> pd.DataFrame:
    return annotate_vep(df, max_variants=100)


@st.cache_data(show_spinner=False)
def _cached_annotate_gnomad(df: pd.DataFrame) -> pd.DataFrame:
    return annotate_gnomad(df, max_variants=50)


@st.cache_data(show_spinner=False)
def _cached_annotate_genes(df: pd.DataFrame) -> pd.DataFrame:
    return annotate_with_genes(df)


def _load_with_validation(vcf_file_or_path, label: str = "VCF") -> pd.DataFrame:
    ok, err = validate_vcf(vcf_file_or_path)
    if not ok:
        st.error(f"❌ **{label} validation failed:** {err}")
        log.error("VCF validation failed for %s: %s", label, err)
        st.stop()
    try:
        if hasattr(vcf_file_or_path, "read"):
            data = vcf_file_or_path.read()
            if hasattr(vcf_file_or_path, "seek"):
                vcf_file_or_path.seek(0)
            return _cached_load_bytes(data, vcf_file_or_path.name)
        return _cached_load_path(str(vcf_file_or_path))
    except Exception as exc:
        st.error(f"❌ Failed to parse {label}: {exc}")
        log.exception("Parse error for %s", label)
        st.stop()


def _safe_load(uploaded, use_ex: bool, ex_path: str, label: str):
    if uploaded:
        return _load_with_validation(uploaded, label)
    if use_ex:
        return _load_with_validation(ex_path, label)
    return None


def _chrom_sort_key(c):
    s = str(c).lower().lstrip("chr")
    return (0, int(s)) if s.isdigit() else (1, s)


def _omim_link(gene: str) -> str:
    if gene and gene not in ("", ".", "—"):
        return f"https://omim.org/search?search={gene}"
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — branding + mode selector
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: .5rem 0 1rem;">
      <div style="font-size:2.4rem">🧬</div>
      <div style="color:#93c5fd; font-size:1.1rem; font-weight:700; letter-spacing:.5px">
        VARIANT ANALYSIS SUITE</div>
      <div style="color:#64748b; font-size:.75rem; margin-top:.2rem">v2.0 · Industry Edition</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    MODE_ICONS = {
        "🔬 Single VCF": "🔬 Single VCF",
        "⚖️ Multi-VCF Compare": "⚖️ Multi-VCF Compare",
        "👨‍👩‍👧 Trio Analysis": "👨‍👩‍👧 Trio Analysis",
        "🧫 Somatic (Tumor/Normal)": "🧫 Somatic (Tumor/Normal)",
        "📦 Batch Pipeline": "📦 Batch Pipeline",
    }
    mode = st.radio(
        "**Analysis Mode**",
        list(MODE_ICONS.keys()),
        label_visibility="visible",
    )
    st.divider()


EX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example.vcf")
EX_ANNOTATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example_annotated.vcf")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE VCF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "🔬 Single VCF":

    with st.sidebar:
        with st.expander("📂 Data Input", expanded=True):
            vcf_file   = st.file_uploader("Upload VCF", type=["vcf", "vcf.gz"],
                                          help="Supports plain or gzipped VCF files up to 500 MB")
            use_example = st.checkbox("Use example VCF", value=True)
            use_annotated = st.checkbox("Use annotated example (SnpEff + ClinVar)", value=False,
                                        help="Loads a VCF pre-annotated with SnpEff ANN and ClinVar CLNSIG fields — enables the SnpEff and ClinVar tabs")

        _ex = EX_ANNOTATED_PATH if (use_annotated and not vcf_file) else EX_PATH
        df_raw = _safe_load(vcf_file, use_example or use_annotated, _ex, "VCF")
        if df_raw is None or df_raw.empty or "chrom" not in df_raw.columns:
            st.title("🧬 Variant Analysis Suite")
            st.info("Upload a VCF file or enable **Use example VCF** to begin.")
            st.stop()

        with st.expander("🔧 Variant Filters", expanded=True):
            min_quality = st.slider("Min Quality (QUAL)", 0, 100, DEFAULT_MIN_QUAL,
                                    help="Minimum Phred-scaled quality score")
            min_depth   = st.slider("Min Depth (DP)", 0, 500, DEFAULT_MIN_DP,
                                    help="Minimum read depth at the variant site")
            all_chroms  = sorted(df_raw["chrom"].unique().tolist(), key=_chrom_sort_key)
            sel_chroms  = st.multiselect(f"Chromosomes ({len(all_chroms)})", all_chroms, default=[])
            det_types   = sorted(df_raw["variant_type"].dropna().unique().tolist())
            var_type    = st.selectbox("Variant Type", ["All"] + det_types)
            pass_only   = st.checkbox("PASS variants only", help="Only include variants with FILTER=PASS")
            af_range    = (
                st.slider("AF Range", 0.0, 1.0, (0.0, 1.0), step=0.01,
                          help="Allele Frequency range filter")
                if "af" in df_raw.columns and df_raw["af"].notna().any()
                else (None, None)
            )

        with st.expander("🧬 Gene Panel Filter", expanded=False):
            panel_choice  = st.selectbox("Built-in Panel", list_panels())
            custom_panel  = st.file_uploader("Or upload gene list (.txt / .csv)", type=["txt","csv"],
                                             key="panel_upload")
            apply_panel   = st.checkbox("Apply panel filter", value=False)

        with st.expander("🔬 Annotations", expanded=False):
            do_ensembl = st.checkbox("Gene names (Ensembl)", help="Queries Ensembl REST API")
            do_vep     = st.checkbox("VEP consequences (first 100)", help="SIFT, PolyPhen, HGVS")
            do_gnomad  = st.checkbox("gnomAD population AF (first 50)")
            do_scores  = st.checkbox("Predictor scores (CADD, SpliceAI, REVEL)")
            do_acmg    = st.checkbox("ACMG-lite classification")
            do_priority = st.checkbox("Variant prioritization score")
            st.caption("⚠️ API annotations require internet; slow for large VCFs")

    # ── Apply filters ─────────────────────────────────────────────────────────
    df = apply_filters(
        df_raw, min_quality=min_quality, min_depth=min_depth,
        variant_type=var_type, chromosomes=sel_chroms if sel_chroms else None,
        min_af=af_range[0], max_af=af_range[1], filter_pass_only=pass_only,
    )

    # Gene panel filter
    if apply_panel:
        if panel_choice == "Custom upload" and custom_panel:
            genes = parse_custom_panel(custom_panel)
        elif panel_choice != "Custom upload":
            genes = get_panel_genes(panel_choice)
        else:
            genes = []
        if genes:
            df = filter_to_panel(df, genes)
            st.sidebar.success(f"Panel: {len(genes)} genes → {len(df)} variants")

    # Annotations
    if do_ensembl and not df.empty:
        with st.spinner("Querying Ensembl for gene names…"):
            df = _cached_annotate_genes(df)
    if do_vep and not df.empty:
        with st.spinner("Running VEP annotation (first 100 variants)…"):
            df = _cached_annotate_vep(df)
    if do_gnomad and not df.empty:
        with st.spinner("Querying gnomAD (first 50 variants)…"):
            df = _cached_annotate_gnomad(df)
    if do_scores and not df.empty:
        with st.spinner("Parsing predictor scores from INFO…"):
            df = parse_predictor_scores(df)
    if do_acmg and not df.empty:
        with st.spinner("Running ACMG-lite classification…"):
            df = classify_dataframe(df)
    if do_priority and not df.empty:
        with st.spinner("Computing prioritization scores…"):
            df = prioritize_dataframe(df)

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🧬 Variant Analysis Suite")
    sample_cols = [c for c in df_raw.columns if c.startswith("sample_")]
    samples     = [c.replace("sample_", "").replace("_GT", "") for c in sample_cols]
    fname       = vcf_file.name if vcf_file else "example.vcf"
    sub_parts   = [f"📄 `{fname}`"]
    if samples:
        sub_parts.append(f"👥 Samples: **{', '.join(samples)}**")
    st.caption("  ·  ".join(sub_parts))

    pass_rate = round(len(df) / len(df_raw) * 100, 1) if len(df_raw) > 0 else 0
    stats = variant_stats(df)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total", f"{len(df_raw):,}", help="All variants in VCF")
    c2.metric("Passing", f"{len(df):,}", help="After applied filters")
    c3.metric("Pass Rate", f"{pass_rate}%")
    c4.metric("Ts/Tv", stats.get("tstv_ratio", "—"), help="Transition/Transversion ratio (WGS expect ~2.1)")
    c5.metric("Mean Depth", stats.get("mean_depth", "—"))
    c6.metric("Chromosomes", df["chrom"].nunique() if "chrom" in df.columns else 0)

    st.divider()

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_names = [
        "📊 Overview", "📈 Distributions", "🗺️ Genome Browser", "👥 Multi-Sample",
        "🎯 Prioritize", "🧬 Gene Panel", "🔬 VEP", "🧪 SnpEff", "🏥 ClinVar",
        "🧬 ACMG", "📉 Statistics", "📋 Predictors", "📋 Data Table", "📄 Report",
    ]
    tabs = st.tabs(tab_names)

    # ── 0: Overview ───────────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(chromosome_plot(df), width="stretch")
        c2.plotly_chart(variant_type_plot(df), width="stretch")
        c3, c4 = st.columns(2)
        c3.plotly_chart(quality_distribution(df), width="stretch")
        c4.plotly_chart(tstv_plot(df), width="stretch")

    # ── 1: Distributions ──────────────────────────────────────────────────────
    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(depth_distribution(df), width="stretch")
        if "af" in df.columns and df["af"].notna().any():
            c2.plotly_chart(af_scatter(df), width="stretch")
        else:
            c2.info("AF data not available in this VCF.")

    # ── 2: Genome Browser ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header">Positional Variant Track</div>', unsafe_allow_html=True)
        if not df.empty and "chrom" in df.columns:
            chrom_sel = st.selectbox("Chromosome", sorted(df["chrom"].unique().tolist(), key=_chrom_sort_key))
            st.plotly_chart(positional_track(df, chrom_sel), width="stretch")
            st.divider()
            st.markdown('<div class="section-header">IGV Genome Browser</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                chrom_clean = chrom_sel.replace("chr", "")
                sub_df = df[df["chrom"] == chrom_sel]
                if not sub_df.empty:
                    first_pos = int(sub_df["position"].iloc[0])
                    igv_url = f"https://igv.org/app/#locus=chr{chrom_clean}:{max(1, first_pos-500)}-{first_pos+500}"
                    st.link_button("🔗 Open region in IGV Web App", igv_url, type="primary")
                    st.caption(f"Opens `chr{chrom_clean}:{first_pos}` in a new tab — free, no install needed.")
            with col2:
                st.info("**IGV embed note**\n\nFor a fully embedded IGV viewer, deploy locally and use `st.components.v1.html()` with IGV.js — not available on Streamlit Cloud due to CORS.")
        else:
            st.info("No variants to display.")

    # ── 3: Multi-Sample ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="section-header">Per-Sample Genotypes</div>', unsafe_allow_html=True)
        if sample_cols:
            disp = ["chrom","position","ref","alt","variant_type","quality","depth"] + sample_cols
            st.dataframe(df[[c for c in disp if c in df.columns]], width="stretch", height=380)
            st.markdown('<div class="section-header">Genotype Distribution per Sample</div>', unsafe_allow_html=True)
            for col in sample_cols:
                sname = col.replace("sample_","").replace("_GT","")
                counts = df[col].value_counts().reset_index()
                counts.columns = ["Genotype","Count"]
                fig = px.bar(counts, x="Genotype", y="Count",
                             title=f"{sname} — Genotype Distribution",
                             color="Genotype")
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No sample genotype columns found in this VCF.")

    # ── 4: Prioritization ─────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header">🎯 Variant Prioritization</div>', unsafe_allow_html=True)
        if "priority_score" not in df.columns:
            st.info("Enable **Variant prioritization score** in the sidebar to rank variants.")
        else:
            tier_counts = df["priority_tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier","Count"]
            c1, c2 = st.columns([1, 2])
            with c1:
                for _, row in tier_counts.iterrows():
                    css = "tier-high" if "HIGH" in row["Tier"] else ("tier-medium" if "MEDIUM" in row["Tier"] else "tier-low")
                    st.markdown(f'<div class="{css}">{row["Tier"]}: {row["Count"]} variants</div>', unsafe_allow_html=True)
            with c2:
                fig = px.bar(tier_counts, x="Tier", y="Count", color="Tier",
                             color_discrete_map={"🔴 HIGH":"#dc2626","🟠 MEDIUM":"#ea580c","🟢 LOW":"#16a34a"},
                             title="Priority Tier Distribution")
                st.plotly_chart(fig, width="stretch")

            st.divider()
            st.markdown("**Top 20 highest-priority variants**")
            priority_cols = ["chrom","position","ref","alt","variant_type","quality","depth",
                             "priority_score","priority_tier","score_breakdown"]
            disp = df[[c for c in priority_cols if c in df.columns]].head(20)
            st.dataframe(disp, width="stretch")
            st.download_button("⬇️ Download Prioritized Variants (CSV)",
                               df[[c for c in priority_cols if c in df.columns]].to_csv(index=False).encode(),
                               "prioritized_variants.csv", "text/csv")

    # ── 5: Gene Panel ─────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-header">🧬 Gene Panel Explorer</div>', unsafe_allow_html=True)
        for pname, pgenes in sorted(
            {k: v for k, v in __import__("utils.gene_panel", fromlist=["PANELS"]).PANELS.items()}.items()
        ):
            with st.expander(f"📋 {pname} ({len(pgenes)} genes)"):
                gene_list_str = "  ·  ".join(sorted(pgenes))
                st.markdown(f"<small>{gene_list_str}</small>", unsafe_allow_html=True)
                if apply_panel and panel_choice == pname:
                    st.success(f"✅ Active — filtering to {len(df)} variants")

        if apply_panel and not df.empty:
            st.divider()
            st.markdown(f"**Variants in active panel ({len(df)} total)**")
            gene_col = next((c for c in ["gene_name","vep_symbol","gene"] if c in df.columns), None)
            if gene_col:
                gene_counts = df[gene_col].value_counts().head(20).reset_index()
                gene_counts.columns = ["Gene","Count"]
                fig = px.bar(gene_counts, x="Gene", y="Count", title="Variants per Panel Gene")
                st.plotly_chart(fig, width="stretch")
                # OMIM links
                st.markdown("**OMIM links for top genes:**")
                for gene in gene_counts["Gene"].head(10):
                    url = _omim_link(str(gene))
                    if url:
                        st.markdown(f"[{gene}]({url})", unsafe_allow_html=False)

    # ── 6: VEP ────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-header">🔬 Ensembl VEP Annotation</div>', unsafe_allow_html=True)
        vep_cols = [c for c in df.columns if c.startswith("vep_")]
        if not vep_cols:
            st.info("Enable **VEP consequences** in the sidebar to annotate variants.\n\n"
                    "Provides: consequence terms, SIFT/PolyPhen predictions, HGVS notation, "
                    "rsIDs, canonical transcript IDs.")
        else:
            disp_cols = ["chrom","position","ref","alt","variant_type"] + vep_cols
            vep_df = df[[c for c in disp_cols if c in df.columns]]
            # Check if all VEP results are empty (API may have failed)
            non_empty = vep_df[vep_cols].astype(bool).any(axis=1).sum()
            if non_empty == 0:
                st.warning("⚠️ VEP returned no annotations. This can happen if:\n"
                           "- Chromosome names use 'chr' prefix but Ensembl expects plain numbers\n"
                           "- The Ensembl API is temporarily unavailable\n"
                           "- Variants are on unrecognised contigs (e.g. alt chromosomes)\n\n"
                           "Try again or check the [Ensembl REST API status](https://rest.ensembl.org).")
            else:
                st.dataframe(vep_df, width="stretch", height=400)

                if "vep_impact" in df.columns:
                    impact_counts = df["vep_impact"][df["vep_impact"] != ""].value_counts().reset_index()
                    impact_counts.columns = ["Impact","Count"]
                    if not impact_counts.empty:
                        fig = px.bar(impact_counts, x="Impact", y="Count", color="Impact",
                                     color_discrete_map=IMPACT_COLORS, title="VEP Impact Distribution")
                        st.plotly_chart(fig, width="stretch")

                if "vep_symbol" in df.columns:
                    top_genes = df["vep_symbol"][df["vep_symbol"] != ""].value_counts().head(15).reset_index()
                    top_genes.columns = ["Gene","Count"]
                    if not top_genes.empty:
                        fig2 = px.bar(top_genes, x="Gene", y="Count", title="Top Affected Genes (VEP)")
                        st.plotly_chart(fig2, width="stretch")

                st.download_button("⬇️ Download VEP Annotations (CSV)",
                                   vep_df.to_csv(index=False).encode(),
                                   "vep_annotations.csv", "text/csv")

    # ── 7: SnpEff ─────────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown('<div class="section-header">🧪 SnpEff Functional Annotation</div>', unsafe_allow_html=True)
        ann_df = parse_snpeff(df)
        if ann_df.empty:
            st.info("No SnpEff ANN field found in this VCF.\n\n"
                    "**To use this feature**, your VCF must be pre-annotated with SnpEff:\n"
                    "```\nsnpEff ann GRCh38.86 input.vcf > annotated.vcf\n```\n"
                    "Or download the [example annotated VCF](data/example_annotated.vcf) to try the feature immediately.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                imp = impact_summary(ann_df)
                fig = px.bar(imp, x="Impact", y="Count", color="Impact",
                             color_discrete_map=IMPACT_COLORS, title="Variants by Impact Level")
                st.plotly_chart(fig, width="stretch")
            with c2:
                genes = top_affected_genes(ann_df)
                fig2  = px.bar(genes.head(15), x="Gene", y="Count", color="High Impact",
                               title="Most Frequently Affected Genes")
                st.plotly_chart(fig2, width="stretch")
            st.dataframe(ann_df, width="stretch", height=380)
            st.download_button("⬇️ Download SnpEff Annotations (CSV)",
                               ann_df.to_csv(index=False).encode(),
                               "snpeff_annotations.csv", "text/csv")

    # ── 8: ClinVar ────────────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown('<div class="section-header">🏥 ClinVar Clinical Significance</div>', unsafe_allow_html=True)
        clin_df = clinvar_significance(df)
        if clin_df.empty or clin_df["ClinVar Significance"].eq("Unknown").all():
            st.info("No ClinVar CLNSIG field found in this VCF.\n\n"
                    "**To use this feature**, annotate your VCF with ClinVar:\n"
                    "```\nbcftools annotate -a clinvar.vcf.gz -c INFO/CLNSIG,INFO/CLNDN input.vcf\n```\n"
                    "Or download the [example annotated VCF](data/example_annotated.vcf) to try the feature immediately.")
        else:
            sig_counts = clin_df["ClinVar Significance"].value_counts().reset_index()
            sig_counts.columns = ["Significance","Count"]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(sig_counts, names="Significance", values="Count",
                                   title="ClinVar Significance Distribution"),
                            width="stretch")
            pathogenic = clin_df[clin_df["ClinVar Significance"].str.contains("Pathogenic", case=False, na=False)]
            if not pathogenic.empty:
                with c2:
                    st.markdown(f"**⚠️ {len(pathogenic)} Pathogenic / Likely Pathogenic variants**")
                    st.dataframe(pathogenic, width="stretch")

    # ── 9: ACMG ───────────────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown('<div class="section-header">🧬 ACMG-lite Classification</div>', unsafe_allow_html=True)
        if "acmg_class" not in df.columns:
            st.info("Enable **ACMG-lite classification** in the sidebar.")
        else:
            COLOR_MAP = {"Pathogenic":"#dc2626","Likely Pathogenic":"#ea580c",
                         "VUS":"#ca8a04","Likely Benign":"#16a34a","Benign":"#0284c7"}
            acmg_counts = df["acmg_class"].value_counts().reset_index()
            acmg_counts.columns = ["Classification","Count"]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(acmg_counts, names="Classification", values="Count",
                                   color="Classification", color_discrete_map=COLOR_MAP,
                                   title="ACMG Classification Distribution"),
                            width="stretch")
            c2.plotly_chart(px.bar(acmg_counts, x="Classification", y="Count",
                                   color="Classification", color_discrete_map=COLOR_MAP,
                                   title="ACMG Classification Counts"),
                            width="stretch")
            st.warning("⚠️ ACMG-lite is a triage tool only. Not for clinical decision-making. "
                       "Confirm with [VarSome](https://varsome.com) or [InterVar](http://www.intervar.org/).")
            acmg_disp = df[["chrom","position","ref","alt","variant_type",
                             "acmg_class","acmg_path_evidence","acmg_benign_evidence"]].copy()
            st.dataframe(acmg_disp, width="stretch", height=380)
            st.download_button("⬇️ Download ACMG Classifications (CSV)",
                               acmg_disp.to_csv(index=False).encode(),
                               "acmg_classifications.csv", "text/csv")

    # ── 10: Statistics ────────────────────────────────────────────────────────
    with tabs[10]:
        st.markdown('<div class="section-header">📉 Comprehensive QC Statistics</div>', unsafe_allow_html=True)
        s = stats
        c1,c2,c3 = st.columns(3)
        c1.metric("SNPs", s.get("snp_count","—"))
        c2.metric("INDELs", s.get("indel_count","—"))
        c3.metric("MNPs", s.get("mnp_count","—"))
        c1.metric("Transitions (Ts)", s.get("ts_count","—"))
        c2.metric("Transversions (Tv)", s.get("tv_count","—"))
        c3.metric("Ts/Tv Ratio", s.get("tstv_ratio","—"),
                  help="Expected: ~2.1 for WGS SNPs, ~3.0 for WES SNPs")
        if s.get("het") is not None:
            c1.metric("Het (avg/sample)", s["het"])
            c2.metric("Hom Alt (avg/sample)", s["hom_alt"])
            c3.metric("Het/Hom Ratio", s.get("het_hom_ratio","—"),
                      help="Expected: ~2.0–2.5 for WGS. Very high may indicate contamination.")
            c1.metric("Missingness %", s.get("missingness_pct","—"))
        c1.metric("Mean QUAL", s.get("mean_qual","—"))
        c2.metric("Median QUAL", s.get("median_qual","—"))
        c3.metric("Mean Depth", s.get("mean_depth","—"))
        st.divider()
        dpc = depth_per_chrom(df)
        if not dpc.empty:
            st.markdown("**Read Depth per Chromosome**")
            st.plotly_chart(px.bar(dpc, x="Chromosome", y="Mean Depth",
                                   hover_data=["Median Depth","Variant Count"],
                                   title="Mean Read Depth per Chromosome"),
                            width="stretch")
            st.dataframe(dpc, width="stretch")

    # ── 11: Predictor Scores ──────────────────────────────────────────────────
    with tabs[11]:
        st.markdown('<div class="section-header">📋 Pathogenicity Predictor Scores</div>', unsafe_allow_html=True)
        score_cols = [c for c in ["cadd_phred","revel_score","spliceai_max_delta",
                                  "alphamissense_score","alphamissense_class"] if c in df.columns]
        if not score_cols:
            st.info("Enable **Predictor scores** in the sidebar to parse CADD, SpliceAI, "
                    "REVEL, and AlphaMissense scores from the INFO field.\n\n"
                    "These are present in VCFs annotated with bcftools/dbNSFP/SpliceAI pipelines.")
        else:
            summary = score_summary(df)
            if not summary.empty:
                st.dataframe(summary, width="stretch")
                st.divider()
            for col, title, threshold in [
                ("cadd_phred",   "CADD Phred Score Distribution", 20),
                ("revel_score",  "REVEL Score Distribution",      0.5),
                ("spliceai_max_delta", "SpliceAI Max Delta Score", 0.5),
                ("alphamissense_score","AlphaMissense Score",      0.5),
            ]:
                if col in df.columns and df[col].notna().any():
                    fig = px.histogram(df, x=col, nbins=50, title=title)
                    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                                  annotation_text=f"Threshold: {threshold}")
                    st.plotly_chart(fig, width="stretch")

    # ── 12: Data Table ────────────────────────────────────────────────────────
    with tabs[12]:
        st.markdown('<div class="section-header">📋 Filtered Variants</div>', unsafe_allow_html=True)
        display_df = df.drop(columns=["info_raw"], errors="ignore")
        st.dataframe(display_df, width="stretch", height=440)
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Download CSV", display_df.to_csv(index=False).encode(),
                           "filtered_variants.csv", "text/csv")
        vcf_lines = ["##fileformat=VCFv4.2",
                     "##source=VariantAnalysisSuite",
                     "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
        for _, row in display_df.iterrows():
            vcf_lines.append(
                f"{row.get('chrom','.')}\t{row.get('position','.')}\t.\t"
                f"{row.get('ref','.')}\t{row.get('alt','.')}\t"
                f"{row.get('quality','.')}\t{row.get('filter','PASS')}\t"
                f"DP={row.get('depth',0)}"
            )
        c2.download_button("⬇️ Download VCF", "\n".join(vcf_lines).encode(),
                           "filtered_variants.vcf", "text/plain")

    # ── 13: Report ────────────────────────────────────────────────────────────
    with tabs[13]:
        st.markdown('<div class="section-header">📄 Export Reports</div>', unsafe_allow_html=True)
        fname_clean = (vcf_file.name if vcf_file else "example.vcf").replace(" ","_")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**HTML Report**\nSelf-contained, shareable with colleagues")
            if st.button("🔄 Generate HTML Report", type="primary"):
                with st.spinner("Building HTML report…"):
                    html_bytes = generate_report(df_raw, df, stats, filename=fname_clean)
                st.download_button("⬇️ Download HTML Report", html_bytes,
                                   fname_clean.replace(".vcf","") + "_report.html", "text/html")
        with c2:
            st.markdown("**PDF Report**\nProfessional format for clinical/lab use")
            if not pdf_available():
                st.warning("Install `fpdf2` to enable PDF export: `pip install fpdf2`")
            elif st.button("🔄 Generate PDF Report"):
                with st.spinner("Building PDF report…"):
                    pdf_bytes = generate_pdf(df_raw, df, stats, filename=fname_clean)
                if pdf_bytes:
                    st.download_button("⬇️ Download PDF Report", pdf_bytes,
                                       fname_clean.replace(".vcf","") + "_report.pdf", "application/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — MULTI-VCF COMPARE (up to 10 files)
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "⚖️ Multi-VCF Compare":

    with st.sidebar:
        with st.expander("📂 Upload VCF Files", expanded=True):
            n_vcfs = st.slider("Number of VCFs to compare", 2, 10, 2)
            uploaded = []
            for i in range(n_vcfs):
                f = st.file_uploader(f"VCF {chr(65+i)}", type=["vcf","vcf.gz"], key=f"cmp_{i}")
                uploaded.append(f)
            use_demo = st.checkbox("Use example VCF for all slots", value=True)

    st.title("⚖️ Multi-VCF Comparison")

    # Load all VCFs
    dfs = []
    names = []
    for i, f in enumerate(uploaded):
        lbl = chr(65 + i)
        if f:
            df_i = _load_with_validation(f, label=f"VCF {lbl}")
            dfs.append(df_i)
            names.append(f.name)
        elif use_demo:
            df_i = _load_with_validation(EX_PATH, label=f"VCF {lbl} (example)")
            dfs.append(df_i)
            names.append(f"Example-{lbl}")

    if len(dfs) < 2:
        st.info("Upload at least 2 VCF files (or enable demo mode) to compare.")
        st.stop()

    # Pairwise comparison matrix
    st.subheader(f"📊 Pairwise Comparison — {len(dfs)} VCFs")
    n = len(dfs)

    # Summary row
    cols = st.columns(n + 1)
    cols[0].markdown("**VCF**")
    for i, name in enumerate(names):
        cols[i+1].markdown(f"**{chr(65+i)}: {name[:18]}**")

    # Concordance heatmap data
    conc_matrix = []
    for i in range(n):
        row_data = []
        for j in range(n):
            if i == j:
                row_data.append(100.0)
            else:
                res = compare_vcfs(dfs[i], dfs[j])
                row_data.append(res["concordance"])
        conc_matrix.append(row_data)

    import plotly.graph_objects as go
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=conc_matrix,
        x=[chr(65+i) for i in range(n)],
        y=[chr(65+i) for i in range(n)],
        colorscale="Blues",
        zmin=0, zmax=100,
        text=[[f"{v:.1f}%" for v in row] for row in conc_matrix],
        texttemplate="%{text}",
        colorbar=dict(title="Concordance %"),
    ))
    heatmap_fig.update_layout(title="Pairwise Concordance Heatmap")
    st.plotly_chart(heatmap_fig, width="stretch")

    # Per-VCF summary
    st.divider()
    st.subheader("Per-VCF Summary")
    summary_rows = []
    for i, (df_i, name) in enumerate(zip(dfs, names)):
        s = variant_stats(df_i)
        summary_rows.append({
            "VCF": f"{chr(65+i)}: {name}",
            "Total Variants": len(df_i),
            "SNPs": s.get("snp_count","—"),
            "INDELs": s.get("indel_count","—"),
            "Ts/Tv": s.get("tstv_ratio","—"),
            "Mean QUAL": s.get("mean_qual","—"),
            "Mean Depth": s.get("mean_depth","—"),
            "Chromosomes": df_i["chrom"].nunique() if "chrom" in df_i.columns else 0,
        })
    st.dataframe(pd.DataFrame(summary_rows), width="stretch")

    # Side-by-side plots
    st.divider()
    st.subheader("Side-by-Side Variant Type Distribution")
    plot_cols = st.columns(min(n, 4))
    for i, df_i in enumerate(dfs[:4]):
        plot_cols[i].plotly_chart(variant_type_plot(df_i), width="stretch")
        plot_cols[i].caption(names[i][:25])

    # Pairwise detail tabs (first pair)
    st.divider()
    st.subheader("Detailed Pairwise Comparison")
    pair_a = st.selectbox("VCF A", range(n), format_func=lambda i: f"{chr(65+i)}: {names[i]}")
    pair_b = st.selectbox("VCF B", range(n), index=min(1, n-1), format_func=lambda i: f"{chr(65+i)}: {names[i]}")
    if pair_a != pair_b:
        result = compare_vcfs(dfs[pair_a], dfs[pair_b])
        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"{chr(65+pair_a)} only", result["n_only_a"])
        c2.metric("Shared", result["n_shared"])
        c3.metric(f"{chr(65+pair_b)} only", result["n_only_b"])
        c4.metric("Concordance", f"{result['concordance']}%")
        st.progress(result["concordance"] / 100)

        ctabs = st.tabs(["Shared", f"Only {chr(65+pair_a)}", f"Only {chr(65+pair_b)}", "By Type"])
        with ctabs[0]:
            st.dataframe(result["shared"].drop(columns=["info_raw"], errors="ignore"),
                         width="stretch", height=350)
            st.download_button("⬇️ Shared CSV",
                               result["shared"].drop(columns=["info_raw"], errors="ignore").to_csv(index=False).encode(),
                               "shared_variants.csv", "text/csv")
        with ctabs[1]:
            st.dataframe(result["only_a"].drop(columns=["info_raw"], errors="ignore"),
                         width="stretch", height=350)
        with ctabs[2]:
            st.dataframe(result["only_b"].drop(columns=["info_raw"], errors="ignore"),
                         width="stretch", height=350)
        with ctabs[3]:
            conc_by_type = concordance_by_type(dfs[pair_a], dfs[pair_b])
            st.dataframe(conc_by_type, width="stretch")
            st.plotly_chart(px.bar(conc_by_type, x="Variant Type", y="Concordance (%)",
                                   color="Variant Type", title="Concordance by Variant Type"),
                            width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — TRIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "👨‍👩‍👧 Trio Analysis":

    with st.sidebar:
        with st.expander("📂 Upload Trio VCFs", expanded=True):
            f_proband = st.file_uploader("👶 Proband (affected)", type=["vcf","vcf.gz"], key="trio_prob")
            f_mother  = st.file_uploader("👩 Mother",              type=["vcf","vcf.gz"], key="trio_mom")
            f_father  = st.file_uploader("👨 Father",              type=["vcf","vcf.gz"], key="trio_dad")
            use_demo  = st.checkbox("Use example for all (demo)", value=True)

    st.title("👨‍👩‍👧 Trio Analysis — De Novo & Recessive Variant Detection")
    st.info("Upload VCFs for proband + both parents to identify **de novo**, "
            "**homozygous recessive**, and **compound heterozygous** variants.")

    prob = _safe_load(f_proband, use_demo, EX_PATH, "Proband")
    mom  = _safe_load(f_mother,  use_demo, EX_PATH, "Mother")
    dad  = _safe_load(f_father,  use_demo, EX_PATH, "Father")

    if any(v is None for v in [prob, mom, dad]):
        st.stop()

    with st.spinner("Running trio analysis…"):
        trio_result = run_trio_analysis(prob, mom, dad)

    c1, c2, c3 = st.columns(3)
    c1.metric("De Novo Variants", trio_result["n_denovo"],
              help="Present in proband, absent in both parents")
    c2.metric("Homozygous Recessive", trio_result["n_hom_rec"],
              help="Hom alt in proband, het in both parents (autosomal recessive)")
    c3.metric("Compound Het", trio_result["n_comp_het"],
              help="Two het variants in same gene from different parents")

    st.divider()
    trio_tabs = st.tabs(["🔴 De Novo", "🟠 Homozygous Recessive", "🟡 Compound Het"])

    with trio_tabs[0]:
        st.subheader(f"De Novo Variants ({trio_result['n_denovo']})")
        if trio_result["de_novo"].empty:
            st.success("No de novo variants detected.")
        else:
            st.dataframe(trio_result["de_novo"].drop(columns=["info_raw"], errors="ignore"),
                         width="stretch")
            st.download_button("⬇️ Download De Novo CSV",
                               trio_result["de_novo"].drop(columns=["info_raw"], errors="ignore").to_csv(index=False).encode(),
                               "denovo_variants.csv", "text/csv")

    with trio_tabs[1]:
        st.subheader(f"Homozygous Recessive ({trio_result['n_hom_rec']})")
        if trio_result["homozygous_recessive"].empty:
            st.success("No homozygous recessive variants detected.")
        else:
            st.dataframe(trio_result["homozygous_recessive"].drop(columns=["info_raw"], errors="ignore"),
                         width="stretch")

    with trio_tabs[2]:
        st.subheader(f"Compound Heterozygous ({trio_result['n_comp_het']})")
        st.caption("Requires gene annotation. Run VEP or SnpEff first and enable in sidebar.")
        if trio_result["compound_het"].empty:
            st.info("No compound het variants detected (or gene annotation not available).")
        else:
            st.dataframe(trio_result["compound_het"].drop(columns=["info_raw"], errors="ignore"),
                         width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 4 — SOMATIC (TUMOR vs NORMAL)
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "🧫 Somatic (Tumor/Normal)":

    with st.sidebar:
        with st.expander("📂 Upload Paired VCFs", expanded=True):
            f_tumor  = st.file_uploader("🔬 Tumor VCF",  type=["vcf","vcf.gz"], key="som_tumor")
            f_normal = st.file_uploader("✅ Normal VCF", type=["vcf","vcf.gz"], key="som_normal")
            use_demo = st.checkbox("Use example for both (demo)", value=True)

    st.title("🧫 Somatic Variant Analysis — Tumor vs Normal")
    st.info("Upload matched tumor and normal VCFs. Variants present **only in the tumor** "
            "are flagged as putative somatic mutations.")

    df_tumor  = _safe_load(f_tumor,  use_demo, EX_PATH, "Tumor")
    df_normal = _safe_load(f_normal, use_demo, EX_PATH, "Normal")

    if df_tumor is None or df_normal is None:
        st.stop()

    result = compare_vcfs(df_tumor, df_normal)
    somatic = result["only_a"]   # in tumor, absent from normal
    germline = result["shared"]  # in both = germline

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tumor Variants", len(df_tumor))
    c2.metric("Normal Variants", len(df_normal))
    c3.metric("Putative Somatic", len(somatic), help="Only in tumor — germline subtracted")
    c4.metric("Germline (shared)", len(germline))

    st.divider()
    som_tabs = st.tabs(["🔴 Somatic Variants", "✅ Germline (shared)", "📊 Comparison"])
    with som_tabs[0]:
        if somatic.empty:
            st.success("No somatic-only variants detected.")
        else:
            disp = somatic.drop(columns=["info_raw"], errors="ignore")
            st.dataframe(disp, width="stretch", height=420)
            c1, c2 = st.columns(2)
            c1.plotly_chart(variant_type_plot(somatic), width="stretch")
            c2.plotly_chart(chromosome_plot(somatic), width="stretch")
            st.download_button("⬇️ Download Somatic Variants (CSV)",
                               disp.to_csv(index=False).encode(),
                               "somatic_variants.csv", "text/csv")
    with som_tabs[1]:
        st.dataframe(germline.drop(columns=["info_raw"], errors="ignore"),
                     width="stretch", height=380)
    with som_tabs[2]:
        conc_by_type = concordance_by_type(df_tumor, df_normal)
        st.dataframe(conc_by_type, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 5 — BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "📦 Batch Pipeline":

    with st.sidebar:
        with st.expander("⚙️ Pipeline Settings", expanded=True):
            min_qual_batch = st.slider("Min Quality", 0, 100, DEFAULT_MIN_QUAL, key="batch_qual")
            min_dp_batch   = st.slider("Min Depth", 0, 500, DEFAULT_MIN_DP,    key="batch_dp")
            do_acmg_batch  = st.checkbox("ACMG-lite classification", key="batch_acmg")
            do_scores_batch = st.checkbox("Parse predictor scores", key="batch_scores")

    st.title("📦 Batch Pipeline")
    st.info("Upload multiple VCF files. All will be filtered with the same settings "
            "and merged into a single annotated CSV download.")

    batch_files = st.file_uploader("Upload VCF files (up to 20)",
                                   type=["vcf","vcf.gz"],
                                   accept_multiple_files=True,
                                   key="batch_upload")

    if not batch_files:
        st.caption("Upload VCF files above to begin batch processing.")
        st.stop()

    if st.button("▶️ Run Batch Pipeline", type="primary"):
        all_dfs = []
        progress = st.progress(0)
        status   = st.empty()

        for i, f in enumerate(batch_files):
            status.text(f"Processing {f.name} ({i+1}/{len(batch_files)})…")
            try:
                df_i = _load_with_validation(f, label=f.name)
                df_i = apply_filters(df_i, min_quality=min_qual_batch,
                                     min_depth=min_dp_batch, variant_type="All")
                if do_scores_batch:
                    df_i = parse_predictor_scores(df_i)
                if do_acmg_batch:
                    df_i = classify_dataframe(df_i)
                df_i["source_file"] = f.name
                all_dfs.append(df_i)
            except Exception as exc:
                st.warning(f"⚠️ Skipped {f.name}: {exc}")
            progress.progress((i + 1) / len(batch_files))

        status.empty()
        progress.empty()

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined = combined.drop(columns=["info_raw"], errors="ignore")
            st.success(f"✅ Processed {len(batch_files)} VCFs → {len(combined):,} total variants")

            # Summary table
            summary = []
            for df_i in all_dfs:
                s = variant_stats(df_i)
                summary.append({
                    "File": df_i["source_file"].iloc[0],
                    "Variants": len(df_i),
                    "SNPs": s.get("snp_count","—"),
                    "INDELs": s.get("indel_count","—"),
                    "Ts/Tv": s.get("tstv_ratio","—"),
                    "Mean QUAL": s.get("mean_qual","—"),
                })
            st.dataframe(pd.DataFrame(summary), width="stretch")

            st.download_button("⬇️ Download Combined CSV",
                               combined.to_csv(index=False).encode(),
                               "batch_combined_variants.csv", "text/csv")
        else:
            st.error("No VCFs were processed successfully.")
