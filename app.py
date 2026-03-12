import os
import streamlit as st
import pandas as pd
import plotly.express as px

from config import DEFAULT_MIN_QUAL, DEFAULT_MIN_DP
from utils.vcf_parser import load_vcf
from utils.filters import apply_filters
from utils.compare import compare_vcfs, concordance_by_type
from utils.snpeff import parse_snpeff, impact_summary, top_affected_genes, IMPACT_COLORS
from utils.stats import variant_stats, depth_per_chrom, clinvar_significance
from utils.validator import validate_vcf
from utils.acmg import classify_dataframe
from utils.gnomad import annotate_gnomad
from utils.report import generate_report
from utils.logger import log
from utils.plots import (
    chromosome_plot, variant_type_plot, quality_distribution,
    depth_distribution, af_scatter, tstv_plot, positional_track, annotate_with_genes,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Variant Analysis Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] p { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #7dd3fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🧬 Variant Analysis Suite")
st.sidebar.markdown("---")

# Mode selector
mode = st.sidebar.radio("Mode", ["🔬 Single VCF Analysis", "⚖️ Compare Two VCFs"])
st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Parsing VCF…")
def _cached_load_path(path: str) -> pd.DataFrame:
    """Load a VCF from a file-system path with Streamlit caching."""
    log.info("Loading VCF from path: %s", path)
    return load_vcf(path)


@st.cache_data(show_spinner="Parsing VCF…")
def _cached_load_bytes(data: bytes, name: str) -> pd.DataFrame:
    """Load a VCF from raw bytes (uploaded file) with Streamlit caching.
    The *name* parameter is only used as a cache key discriminator.
    """
    import io
    log.info("Loading uploaded VCF: %s (%d bytes)", name, len(data))
    return load_vcf(io.BytesIO(data))


def _load_with_validation(vcf_file_or_path, label: str = "VCF") -> pd.DataFrame:
    """Validate then load a VCF, stopping with a user-friendly error on failure."""
    ok, err = validate_vcf(vcf_file_or_path)
    if not ok:
        st.error(f"❌ **{label} validation failed:** {err}")
        log.error("VCF validation failed for %s: %s", label, err)
        st.stop()

    try:
        if hasattr(vcf_file_or_path, "read"):
            # Streamlit UploadedFile — read bytes once, cache by name+size
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


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE VCF MODE
# ─────────────────────────────────────────────────────────────────────────────
if mode == "🔬 Single VCF Analysis":

    st.sidebar.header("📂 Data Input")
    vcf_file = st.sidebar.file_uploader("Upload VCF file", type=["vcf", "vcf.gz"])
    use_example = st.sidebar.checkbox("Use example VCF", value=True)

    ex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example.vcf")
    df_raw = _safe_load(vcf_file, use_example, ex_path, "VCF")

    if df_raw is None or df_raw.empty or "chrom" not in df_raw.columns:
        st.title("🧬 Variant Analysis Suite")
        st.info("Upload a VCF file or enable **Use example VCF** to get started.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Filters")
    min_quality = st.sidebar.slider("Min Quality", 0, 100, DEFAULT_MIN_QUAL,
                                    key="sq_qual", help="QUAL score threshold (0 = no filter)")
    min_depth   = st.sidebar.slider("Min Depth", 0, 500, DEFAULT_MIN_DP,
                                    key="sq_depth", help="Read depth (DP) threshold")
    all_chroms  = sorted(df_raw["chrom"].unique().tolist(), key=_chrom_sort_key)
    sel_chroms  = st.sidebar.multiselect(f"Chromosomes ({len(all_chroms)} found)", all_chroms, default=[],
                                          help="Leave empty = all. 1, chr1, chrX all work.")
    det_types   = sorted(df_raw["variant_type"].dropna().unique().tolist())
    var_type    = st.sidebar.selectbox("Variant Type", ["All"] + det_types)
    pass_only   = st.sidebar.checkbox("PASS only", value=False)
    af_range    = st.sidebar.slider("AF Range", 0.0, 1.0, (0.0, 1.0), step=0.01) \
                  if "af" in df_raw.columns and df_raw["af"].notna().any() else (None, None)

    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Annotation")
    do_ensembl = st.sidebar.checkbox("Gene names (Ensembl)", value=False)
    do_gnomad  = st.sidebar.checkbox("gnomAD population AF (first 50 variants)", value=False)
    do_acmg    = st.sidebar.checkbox("ACMG-lite classification", value=False)
    st.sidebar.caption("⚠️ gnomAD & Ensembl require internet; slow for large VCFs")

    # ── Apply filters ─────────────────────────────────────────────────────────
    df = apply_filters(df_raw, min_quality=min_quality, min_depth=min_depth,
                       variant_type=var_type,
                       chromosomes=sel_chroms if sel_chroms else None,
                       min_af=af_range[0], max_af=af_range[1],
                       filter_pass_only=pass_only)

    if do_ensembl and not df.empty:
        with st.spinner("Querying Ensembl for gene names…"):
            df = annotate_with_genes(df)

    if do_gnomad and not df.empty:
        with st.spinner("Querying gnomAD (first 50 variants)…"):
            df = annotate_gnomad(df, max_variants=50)

    if do_acmg and not df.empty:
        with st.spinner("Running ACMG-lite classification…"):
            df = classify_dataframe(df)

    # ── Header & metrics ──────────────────────────────────────────────────────
    st.title("🧬 Variant Analysis Suite")
    sample_cols = [c for c in df_raw.columns if c.startswith("sample_")]
    samples = [c.replace("sample_", "").replace("_GT", "") for c in sample_cols]
    if samples:
        st.caption(f"Samples: **{', '.join(samples)}**")

    pass_rate = round(len(df) / len(df_raw) * 100, 1) if len(df_raw) > 0 else 0
    stats = variant_stats(df)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total", len(df_raw))
    c2.metric("Passing", len(df))
    c3.metric("Pass Rate", f"{pass_rate}%")
    c4.metric("Ts/Tv", stats.get("tstv_ratio", "—"))
    c5.metric("Mean Depth", stats.get("mean_depth", "—"))
    c6.metric("Chromosomes", df["chrom"].nunique() if "chrom" in df.columns else 0)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "📈 Distributions",
        "🗺️ Genome Browser",
        "👥 Multi-Sample",
        "🧪 SnpEff",
        "🏥 ClinVar",
        "🧬 ACMG",
        "📉 Statistics",
        "📋 Data Table",
        "📄 Report",
    ])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(chromosome_plot(df), use_container_width=True)
        c2.plotly_chart(variant_type_plot(df), use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(quality_distribution(df), use_container_width=True)
        c4.plotly_chart(tstv_plot(df), use_container_width=True)

    # ── Distributions ─────────────────────────────────────────────────────────
    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(depth_distribution(df), use_container_width=True)
        if "af" in df.columns and df["af"].notna().any():
            c2.plotly_chart(af_scatter(df), use_container_width=True)
        else:
            c2.info("AF data not available.")

    # ── Genome Browser ────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("🗺️ Positional Variant Track")
        if not df.empty and "chrom" in df.columns:
            chrom_sel = st.selectbox("Chromosome", sorted(df["chrom"].unique().tolist(), key=_chrom_sort_key))
            st.plotly_chart(positional_track(df, chrom_sel), use_container_width=True)

            # IGV.js embed
            st.markdown("---")
            st.subheader("🔬 IGV Genome Browser")
            st.info("To use the full IGV browser, upload your VCF/BAM to [igv.org/app](https://igv.org/app/) or embed a local IGV server. Below is a direct link launcher:")
            igv_chrom = chrom_sel.replace("chr", "")
            if not df.empty:
                first_pos = int(df[df["chrom"] == chrom_sel]["position"].iloc[0]) if len(df[df["chrom"] == chrom_sel]) > 0 else 1
                igv_url = f"https://igv.org/app/#locus=chr{igv_chrom}:{max(1, first_pos-500)}-{first_pos+500}"
                st.link_button("🔗 Open in IGV Web App", igv_url)
        else:
            st.info("No variants to display.")

    # ── Multi-Sample ──────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("👥 Per-Sample Genotypes")
        if sample_cols:
            disp_cols = ["chrom","position","ref","alt","variant_type","quality","depth"] + sample_cols
            st.dataframe(df[[c for c in disp_cols if c in df.columns]], use_container_width=True, height=400)
            st.markdown("**Genotype counts per sample**")
            for col in sample_cols:
                sname = col.replace("sample_","").replace("_GT","")
                counts = df[col].value_counts().reset_index()
                counts.columns = ["Genotype","Count"]
                fig = px.bar(counts, x="Genotype", y="Count", title=f"{sname} Genotype Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sample columns found.")

    # ── SnpEff ────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("🧪 SnpEff Functional Annotation")
        ann_df = parse_snpeff(df)
        if ann_df.empty:
            st.info("No SnpEff ANN field found in this VCF.\n\nRun SnpEff on your VCF first:\n```\nsnpEff ann GRCh38.86 input.vcf > annotated.vcf\n```")
        else:
            imp = impact_summary(ann_df)
            genes = top_affected_genes(ann_df)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Impact Summary**")
                fig = px.bar(imp, x="Impact", y="Count", color="Impact",
                             color_discrete_map=IMPACT_COLORS, title="Variants by Impact Level")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Top Affected Genes**")
                fig2 = px.bar(genes.head(15), x="Gene", y="Count", color="High Impact",
                              title="Most Frequently Affected Genes")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Full Annotation Table**")
            st.dataframe(ann_df, use_container_width=True, height=400)
            csv = ann_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download SnpEff Annotations (CSV)", csv, "snpeff_annotations.csv", "text/csv")

    # ── ClinVar ───────────────────────────────────────────────────────────────
    with tabs[5]:
        st.subheader("🏥 ClinVar Clinical Significance")
        clin_df = clinvar_significance(df)
        if clin_df.empty or clin_df["ClinVar Significance"].eq("Unknown").all():
            st.info("No ClinVar CLNSIG field found in this VCF.\n\nAnnotate with ClinVar first:\n```\nbcftools annotate -a clinvar.vcf.gz -c INFO/CLNSIG,INFO/CLNDN input.vcf\n```")
        else:
            sig_counts = clin_df["ClinVar Significance"].value_counts().reset_index()
            sig_counts.columns = ["Significance","Count"]
            fig = px.pie(sig_counts, names="Significance", values="Count",
                         title="ClinVar Significance Distribution")
            st.plotly_chart(fig, use_container_width=True)
            pathogenic = clin_df[clin_df["ClinVar Significance"].str.contains("Pathogenic", case=False, na=False)]
            if not pathogenic.empty:
                st.markdown(f"**⚠️ {len(pathogenic)} Pathogenic / Likely Pathogenic variants**")
                st.dataframe(pathogenic, use_container_width=True)

    # ── ACMG ──────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("🧬 ACMG-lite Pathogenicity Classification")
        if "acmg_class" not in df.columns:
            st.info("Enable **ACMG-lite classification** in the sidebar to see results here.")
        else:
            acmg_counts = df["acmg_class"].value_counts().reset_index()
            acmg_counts.columns = ["Classification","Count"]
            COLOR_MAP = {
                "Pathogenic": "#dc2626",
                "Likely Pathogenic": "#ea580c",
                "VUS": "#ca8a04",
                "Likely Benign": "#16a34a",
                "Benign": "#0284c7",
            }
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(acmg_counts, names="Classification", values="Count",
                             color="Classification", color_discrete_map=COLOR_MAP,
                             title="ACMG Classification Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(acmg_counts, x="Classification", y="Count",
                              color="Classification", color_discrete_map=COLOR_MAP,
                              title="ACMG Classification Counts")
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("⚠️ **Disclaimer:** ACMG-lite is a simplified triage tool. It is NOT a "
                        "clinical-grade classifier. Always confirm with [VarSome](https://varsome.com) "
                        "or [InterVar](http://www.intervar.org/) before clinical use.")
            acmg_display = df[["chrom","position","ref","alt","variant_type",
                                "acmg_class","acmg_path_evidence","acmg_benign_evidence"]].copy()
            st.dataframe(acmg_display, use_container_width=True, height=400)
            st.download_button("⬇️ Download ACMG Classifications (CSV)",
                               acmg_display.to_csv(index=False).encode(),
                               "acmg_classifications.csv", "text/csv")

    # ── Statistics ────────────────────────────────────────────────────────────
    with tabs[7]:
        st.subheader("📉 Comprehensive Variant Statistics")
        s = stats
        col1, col2, col3 = st.columns(3)
        col1.metric("SNPs", s.get("snp_count","—"))
        col2.metric("INDELs", s.get("indel_count","—"))
        col3.metric("MNPs", s.get("mnp_count","—"))
        col1.metric("Transitions (Ts)", s.get("ts_count","—"))
        col2.metric("Transversions (Tv)", s.get("tv_count","—"))
        col3.metric("Ts/Tv Ratio", s.get("tstv_ratio","—"))
        if s.get("het") is not None:
            col1.metric("Het (avg/sample)", s["het"])
            col2.metric("Hom Alt (avg/sample)", s["hom_alt"])
            col3.metric("Het/Hom Ratio", s.get("het_hom_ratio","—"))
            col1.metric("Missingness %", s.get("missingness_pct","—"))
        col1.metric("Mean QUAL", s.get("mean_qual","—"))
        col2.metric("Median QUAL", s.get("median_qual","—"))
        col3.metric("Mean Depth", s.get("mean_depth","—"))

        st.markdown("---")
        st.markdown("**Depth per Chromosome**")
        dpc = depth_per_chrom(df)
        if not dpc.empty:
            fig = px.bar(dpc, x="Chromosome", y="Mean Depth",
                         hover_data=["Median Depth","Variant Count"],
                         title="Mean Read Depth per Chromosome")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dpc, use_container_width=True)

    # ── Data Table ────────────────────────────────────────────────────────────
    with tabs[8]:
        st.subheader("📋 Filtered Variants")
        display_df = df.drop(columns=["info_raw"], errors="ignore")
        st.dataframe(display_df, use_container_width=True, height=450)
        csv = display_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv, "filtered_variants.csv", "text/csv")
        vcf_lines = ["##fileformat=VCFv4.2",
                     "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
        for _, row in display_df.iterrows():
            vcf_lines.append(f"{row.get('chrom','.')}\t{row.get('position','.')}\t.\t"
                             f"{row.get('ref','.')}\t{row.get('alt','.')}\t"
                             f"{row.get('quality','.')}\t{row.get('filter','PASS')}\t"
                             f"DP={row.get('depth',0)}")
        st.download_button("⬇️ Download VCF", "\n".join(vcf_lines).encode(),
                           "filtered_variants.vcf", "text/plain")

    # ── Report ────────────────────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("📄 Export Analysis Report")
        st.markdown("Generate a self-contained HTML report you can share with colleagues.")
        fname = (vcf_file.name if vcf_file else "example.vcf").replace(" ", "_")
        if st.button("🔄 Generate Report", type="primary"):
            with st.spinner("Building report…"):
                html_bytes = generate_report(df_raw, df, stats, filename=fname)
            st.success(f"Report ready — {len(df)} filtered variants from {fname}")
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_bytes,
                file_name=fname.replace(".vcf","") + "_report.html",
                mime="text/html",
            )
        else:
            st.info("Click **Generate Report** above to build your report. "
                    "It includes summary metrics, variant type breakdown, "
                    "chromosome distribution, and ACMG classification (if enabled).")



# ─────────────────────────────────────────────────────────────────────────────
# COMPARE TWO VCFs MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.sidebar.header("📂 VCF File A")
    vcf_a = st.sidebar.file_uploader("Upload VCF A", type=["vcf","vcf.gz"], key="vcf_a")
    st.sidebar.header("📂 VCF File B")
    vcf_b = st.sidebar.file_uploader("Upload VCF B", type=["vcf","vcf.gz"], key="vcf_b")
    use_ex = st.sidebar.checkbox("Use example for both (demo)", value=True)

    ex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example.vcf")

    df_a = _safe_load(vcf_a, use_ex, ex_path, "VCF A")
    df_b = _safe_load(vcf_b, use_ex, ex_path, "VCF B")

    if df_a is None or df_b is None:
        st.title("⚖️ VCF Comparison")
        st.info("Upload two VCF files (or enable demo mode) to compare them.")
        st.stop()

    st.title("⚖️ VCF Comparison")
    name_a = vcf_a.name if vcf_a else "Example A"
    name_b = vcf_b.name if vcf_b else "Example B"

    result = compare_vcfs(df_a, df_b)

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(f"{name_a} variants", len(df_a))
    c2.metric(f"{name_b} variants", len(df_b))
    c3.metric("Shared", result["n_shared"])
    c4.metric(f"Only in A", result["n_only_a"])
    c5.metric(f"Only in B", result["n_only_b"])

    st.metric("Concordance", f"{result['concordance']}%")
    st.progress(result["concordance"] / 100)
    st.divider()

    ctabs = st.tabs([
        "📊 Overview",
        "🔬 Shared Variants",
        "🅰️ Unique to A",
        "🅱️ Unique to B",
        "📋 Concordance by Type",
    ])

    with ctabs[0]:
        # Venn-style bar
        venn_data = pd.DataFrame({
            "Category": [f"Only {name_a}", "Shared", f"Only {name_b}"],
            "Count": [result["n_only_a"], result["n_shared"], result["n_only_b"]],
        })
        fig = px.bar(venn_data, x="Category", y="Count",
                     color="Category",
                     color_discrete_sequence=["#1f77b4","#2ca02c","#ff7f0e"],
                     title="Variant Overlap")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{name_a} — Variant Types**")
            st.plotly_chart(variant_type_plot(df_a), use_container_width=True)
        with c2:
            st.markdown(f"**{name_b} — Variant Types**")
            st.plotly_chart(variant_type_plot(df_b), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"**{name_a} — Quality Distribution**")
            st.plotly_chart(quality_distribution(df_a), use_container_width=True)
        with c4:
            st.markdown(f"**{name_b} — Quality Distribution**")
            st.plotly_chart(quality_distribution(df_b), use_container_width=True)

    with ctabs[1]:
        st.subheader(f"✅ {result['n_shared']} Shared Variants")
        shared = result["shared"].drop(columns=["info_raw"], errors="ignore")
        st.dataframe(shared, use_container_width=True, height=400)
        st.download_button("⬇️ Download Shared (CSV)", shared.to_csv(index=False).encode(),
                           "shared_variants.csv", "text/csv")

    with ctabs[2]:
        st.subheader(f"🅰️ {result['n_only_a']} variants only in {name_a}")
        only_a = result["only_a"].drop(columns=["info_raw"], errors="ignore")
        st.dataframe(only_a, use_container_width=True, height=400)
        st.download_button("⬇️ Download Only-A (CSV)", only_a.to_csv(index=False).encode(),
                           f"only_{name_a}.csv", "text/csv")

    with ctabs[3]:
        st.subheader(f"🅱️ {result['n_only_b']} variants only in {name_b}")
        only_b = result["only_b"].drop(columns=["info_raw"], errors="ignore")
        st.dataframe(only_b, use_container_width=True, height=400)
        st.download_button("⬇️ Download Only-B (CSV)", only_b.to_csv(index=False).encode(),
                           f"only_{name_b}.csv", "text/csv")

    with ctabs[4]:
        st.subheader("📋 Concordance Breakdown by Variant Type")
        conc = concordance_by_type(df_a, df_b)
        st.dataframe(conc, use_container_width=True)
        fig = px.bar(conc, x="Variant Type", y="Concordance (%)", color="Variant Type",
                     title="Concordance % per Variant Type")
        st.plotly_chart(fig, use_container_width=True)

