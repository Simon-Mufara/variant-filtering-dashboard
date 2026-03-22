"""Variant Analysis Suite — main Streamlit application."""
import os
import io
import types
import shutil
import streamlit as st
import pandas as pd
import plotly.express as px

from config import DEFAULT_MIN_QUAL, DEFAULT_MIN_DP
from utils import auth as auth_mod
# load_vcf imported via format_parser internally
from utils.validator import validate_vcf
from utils.format_parser import load_any, supported_extensions
from utils.filters import apply_filters
from utils.compare import compare_vcfs, concordance_by_type
from utils.snpeff import parse_snpeff, impact_summary, top_affected_genes, IMPACT_COLORS
from utils.stats import variant_stats, depth_per_chrom, clinvar_significance
from utils.stats import allele_balance_stats, variant_density, missingness_per_sample
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

# Backward-compatible auth bindings (supports older deployed auth.py versions).
require_auth = auth_mod.require_auth
render_user_status = getattr(auth_mod, "render_user_status", lambda _ctx: None)
available_modes = getattr(
    auth_mod,
    "available_modes",
    lambda _role: [
        "🔬 Single VCF",
        "⚖️ Multi-VCF Compare",
        "👨‍👩‍👧 Trio Analysis",
        "🧫 Somatic (Tumor/Normal)",
    ],
)


def _missing_auth_helper(name: str):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"Auth helper '{name}' is unavailable in this deployment. "
            "Please pull latest code for utils/auth.py."
        )

    return _raise


create_user_account = getattr(auth_mod, "create_user_account", _missing_auth_helper("create_user_account"))
create_organization = getattr(auth_mod, "create_organization", _missing_auth_helper("create_organization"))
create_team = getattr(auth_mod, "create_team", _missing_auth_helper("create_team"))
list_users = getattr(auth_mod, "list_users", _missing_auth_helper("list_users"))
list_organizations = getattr(auth_mod, "list_organizations", _missing_auth_helper("list_organizations"))
list_teams = getattr(auth_mod, "list_teams", _missing_auth_helper("list_teams"))
set_user_active = getattr(auth_mod, "set_user_active", _missing_auth_helper("set_user_active"))

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

# ── Auth gate (role-aware auth via Streamlit secrets) ────────────────────────
auth_ctx = require_auth()
if auth_ctx is None:
    auth_ctx = types.SimpleNamespace(
        user_id=0,
        username="anonymous",
        role="individual",
        display_name="Guest",
        organization_name="Independent",
        team_name="N/A",
    )

if "ui_theme_choice" not in st.session_state:
    st.session_state["ui_theme_choice"] = "Light"

_UI_ICONS = {
    "app": "🧬",
    "mode": "🧭",
    "theme": "🎨",
    "workspace": "🏢",
    "data": "📁",
    "filter": "⚙️",
    "annotation": "🧪",
}

_MODE_DESCRIPTIONS = {
    "🔬 Single VCF": "Focused deep-dive for one variant file with full annotation/QC tabs.",
    "⚖️ Multi-VCF Compare": "Pairwise and multi-file concordance review for cohort comparison.",
    "👨‍👩‍👧 Trio Analysis": "Family-based inheritance analysis for de novo and recessive candidates.",
    "🧫 Somatic (Tumor/Normal)": "Tumor-normal subtraction workflow for putative somatic variants.",
    "📦 Batch Pipeline": "Run batch workflows across multiple projects and cases.",
    "🛠️ Admin Console": "Manage organisations, teams, users, and platform governance.",
}


def _resolve_theme_name(choice: str) -> str:
    if choice == "Light":
        return "light"
    if choice == "Dark":
        return "dark"
    try:
        configured = str(st.get_option("theme.base") or "dark").lower()
    except Exception:
        configured = "dark"
    return "light" if configured == "light" else "dark"


def _inject_theme_css(theme_name: str) -> None:
    palette = {
        "light": {
            "page_bg": "linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%)",
            "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%)",
            "sidebar_border": "#e2e8f0",
            "sidebar_text": "#1e293b",
            "sidebar_heading": "#0f172a",
            "sidebar_muted": "#64748b",
            "accent": "#1d4ed8",
            "accent_soft": "linear-gradient(90deg, rgba(59,130,246,.16) 0%, rgba(14,165,233,.16) 100%)",
            "accent_ring": "rgba(37, 99, 235, 0.15)",
            "metric_bg": "#ffffff",
            "metric_border": "#dbe3ef",
            "tabs_bg": "#eaf1ff",
            "tabs_text": "#334155",
            "tabs_active_bg": "#ffffff",
            "tabs_active_text": "#1e40af",
            "divider": "#e2e8f0",
            "section_text": "#0f172a",
            "card_shadow": "0 6px 20px rgba(15, 23, 42, 0.08)",
        },
        "dark": {
            "page_bg": "radial-gradient(circle at top, #0f172a 0%, #020617 56%)",
            "sidebar_bg": "linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)",
            "sidebar_border": "#1f2937",
            "sidebar_text": "#e2e8f0",
            "sidebar_heading": "#bae6fd",
            "sidebar_muted": "#a5b4fc",
            "accent": "#38bdf8",
            "accent_soft": "linear-gradient(90deg, rgba(56,189,248,.2) 0%, rgba(167,139,250,.2) 100%)",
            "accent_ring": "rgba(56, 189, 248, 0.25)",
            "metric_bg": "#111827",
            "metric_border": "#374151",
            "tabs_bg": "#1f2937",
            "tabs_text": "#e5e7eb",
            "tabs_active_bg": "#0f172a",
            "tabs_active_text": "#7dd3fc",
            "divider": "#334155",
            "section_text": "#e2e8f0",
            "card_shadow": "0 10px 24px rgba(2, 6, 23, 0.45)",
        },
    }[theme_name]

    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
          html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
          .block-container {{ padding-top: 0.8rem; padding-bottom: 1rem; }}
          [data-testid="stAppViewContainer"] {{
            background: {palette["page_bg"]};
          }}
          [data-testid="stHeader"] {{
            background: transparent;
          }}

          section[data-testid="stSidebar"] {{
            background: {palette["sidebar_bg"]};
            border-right: 1px solid {palette["sidebar_border"]};
          }}
          section[data-testid="stSidebar"] .stMarkdown,
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] .stSelectbox label,
          section[data-testid="stSidebar"] p,
          section[data-testid="stSidebar"] span {{ color: {palette["sidebar_text"]} !important; }}
          section[data-testid="stSidebar"] h1,
          section[data-testid="stSidebar"] h2,
          section[data-testid="stSidebar"] h3 {{ color: {palette["sidebar_heading"]} !important; }}

          .app-brand {{
            border: 1px solid {palette["sidebar_border"]};
            border-radius: 12px;
            padding: .7rem .8rem;
            background: {palette["accent_soft"]};
            margin-bottom: .5rem;
            box-shadow: {palette["card_shadow"]};
          }}
          .app-brand-icon {{ font-size: 1.8rem; margin-bottom: .2rem; }}
          .app-brand-title {{ font-size: 1rem; font-weight: 700; color: {palette["sidebar_heading"]}; }}
          .app-brand-subtitle {{ font-size: .74rem; color: {palette["sidebar_muted"]}; }}
          .sidebar-note {{
            font-size: .75rem;
            color: {palette["sidebar_muted"]};
            border-top: 1px solid {palette["sidebar_border"]};
            padding-top: .55rem;
            margin-top: .55rem;
            line-height: 1.5;
          }}
          .mode-badge {{
            margin-top: .3rem;
            border: 1px solid {palette["sidebar_border"]};
            border-radius: 8px;
            padding: .45rem .55rem;
            background: {palette["accent_soft"]};
            font-size: .76rem;
            color: {palette["sidebar_text"]};
            box-shadow: 0 0 0 1px {palette["accent_ring"]} inset;
          }}

          [data-testid="metric-container"] {{
            background: {palette["metric_bg"]};
            border: 1px solid {palette["metric_border"]};
            border-radius: 10px;
            padding: 0.8rem 1rem;
            box-shadow: {palette["card_shadow"]};
            backdrop-filter: blur(2px);
          }}
          [data-testid="metric-container"] label {{ color: #64748b; font-size: .8rem; font-weight: 600; letter-spacing: .2px; }}
          [data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {palette["section_text"]}; font-weight: 700; }}

          .stTabs [data-baseweb="tab-list"] {{
            gap: 6px; background: {palette["tabs_bg"]}; border-radius: 10px; padding: 6px;
          }}
          .stTabs [data-baseweb="tab"] {{
            border-radius: 8px; padding: .45rem .95rem; font-size: .85rem; font-weight: 600;
            color: {palette["tabs_text"]};
            border: 1px solid transparent;
          }}
          .stTabs [aria-selected="true"] {{
            background: {palette["tabs_active_bg"]} !important;
            color: {palette["tabs_active_text"]} !important;
            border-color: {palette["accent_ring"]};
            box-shadow: {palette["card_shadow"]};
            font-weight: 600;
          }}

          hr {{ border-color: {palette["divider"]}; }}
          .stAlert {{ border-radius: 8px; }}
          .tier-high {{ color: #dc2626; font-weight: 700; }}
          .tier-medium {{ color: #ea580c; font-weight: 700; }}
          .tier-low {{ color: #16a34a; font-weight: 700; }}
          .section-header {{
            font-size: 1.1rem; font-weight: 700; color: {palette["section_text"]};
            border-left: 4px solid {palette["accent"]}; padding-left: .6rem; border-radius: 2px;
            margin: 1rem 0 .5rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_active_theme_name = _resolve_theme_name(st.session_state["ui_theme_choice"])
_inject_theme_css(_active_theme_name)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Streamlit 1.55 cache hasher walks __globals__ of cached functions and raises
# KeyError when it encounters module-type objects. Bypass with hash_funcs.
_CACHE_HASH_FUNCS = {types.ModuleType: lambda m: m.__name__}


@st.cache_data(show_spinner="Parsing file…", hash_funcs=_CACHE_HASH_FUNCS)
def _cached_load_path(path: str) -> pd.DataFrame:
    log.info("Loading file from path: %s", path)
    return load_any(open(path, "rb"), path)


@st.cache_data(show_spinner="Parsing file…", hash_funcs=_CACHE_HASH_FUNCS)
def _cached_load_bytes(data: bytes, name: str) -> pd.DataFrame:
    log.info("Loading uploaded file: %s (%d bytes)", name, len(data))
    return load_any(io.BytesIO(data), name)


@st.cache_data(show_spinner=False, hash_funcs=_CACHE_HASH_FUNCS)
def _cached_annotate_vep(df: pd.DataFrame, genome_build: str = "GRCh38") -> pd.DataFrame:
    return annotate_vep(df, max_variants=100, genome_build=genome_build)


@st.cache_data(show_spinner=False, hash_funcs=_CACHE_HASH_FUNCS)
def _cached_annotate_gnomad(df: pd.DataFrame, genome_build: str = "GRCh38") -> pd.DataFrame:
    return annotate_gnomad(df, max_variants=50, genome_build=genome_build)


@st.cache_data(show_spinner=False, hash_funcs=_CACHE_HASH_FUNCS)
def _cached_annotate_genes(df: pd.DataFrame) -> pd.DataFrame:
    return annotate_with_genes(df)


@st.cache_data(show_spinner="Downloading VCF from URL…", hash_funcs=_CACHE_HASH_FUNCS)
def _cached_load_url(url: str) -> pd.DataFrame:
    import requests as _requests
    resp = _requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    data = resp.content
    filename = url.split("/")[-1].split("?")[0] or "remote.vcf"
    return load_any(io.BytesIO(data), filename)


def _load_with_validation(file_or_path, label: str = "file") -> pd.DataFrame:
    """Load any supported variant file format with validation."""
    filename = ""
    if hasattr(file_or_path, "name"):
        filename = file_or_path.name
    elif isinstance(file_or_path, str):
        filename = os.path.basename(file_or_path)

    # For uploaded files, read all bytes once upfront so we don't lose position
    if hasattr(file_or_path, "read"):
        raw_bytes = file_or_path.read()
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        # Wrap in BytesIO so both validator and parser can read from the start
        buf = io.BytesIO(raw_bytes)
        buf.name = filename  # validator uses .name if present

        is_vcf = filename.lower().endswith((".vcf", ".vcf.gz")) or not filename
        if is_vcf:
            buf.seek(0)
            ok, err = validate_vcf(buf)
            if not ok:
                st.error(f"❌ **{label} validation failed:** {err}")
                log.error("Validation failed for %s: %s", label, err)
                st.stop()

        try:
            return _cached_load_bytes(raw_bytes, filename)
        except Exception as exc:
            st.error(f"❌ Failed to parse {label}: {exc}")
            log.exception("Parse error for %s", label)
            st.stop()

    # Path-based load
    is_vcf = filename.lower().endswith((".vcf", ".vcf.gz")) or not filename
    if is_vcf:
        ok, err = validate_vcf(file_or_path)
        if not ok:
            st.error(f"❌ **{label} validation failed:** {err}")
            log.error("Validation failed for %s: %s", label, err)
            st.stop()
    try:
        return _cached_load_path(str(file_or_path))
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


def _tool_available(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def _render_tool_status(tool_names: list[str]) -> None:
    rows = []
    for name in tool_names:
        rows.append(
            {
                "Tool": name,
                "Status": "✅ Available" if _tool_available(name) else "❌ Not found",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — branding + mode selector
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"""
        <div class="app-brand">
          <div class="app-brand-icon">{_UI_ICONS["app"]}</div>
          <div class="app-brand-title">Variant Analysis Suite</div>
          <div class="app-brand-subtitle">v3.1 · Clinical Research Workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.selectbox(
        f"{_UI_ICONS['theme']} Interface Theme",
        ["Auto", "Light", "Dark"],
        key="ui_theme_choice",
        help="Use Light for a white interface, Dark for low-light, or Auto to follow app base theme.",
    )
    render_user_status(auth_ctx)
    st.divider()

    with st.expander(f"{_UI_ICONS['workspace']} Workspace Settings", expanded=False):
        st.text_input("Organisation", key="workspace_org", placeholder="e.g. School of Health Sciences")
        st.text_input("Team", key="workspace_team", placeholder="e.g. Cancer Genomics Lab")
        st.text_input("Project", key="workspace_project", placeholder="e.g. Variant Review")
        st.text_input("Case / Batch ID", key="workspace_case_id", placeholder="optional")

    st.divider()

    allowed_modes = available_modes(getattr(auth_ctx, "role", "individual"))
    if not allowed_modes:
        allowed_modes = ["🔬 Single VCF"]
    mode = st.radio(
        f"**{_UI_ICONS['mode']} Analysis Mode**",
        allowed_modes,
        label_visibility="visible",
    )
    st.markdown(
        f'<div class="mode-badge">{_MODE_DESCRIPTIONS.get(mode, "Analysis workflow selected.")}</div>',
        unsafe_allow_html=True,
    )
    if auth_ctx.role == "individual":
        st.caption("Role access: Individual users can run Single VCF workflows.")
    elif auth_ctx.role == "team_member":
        st.caption("Role access: Team users can run collaborative analysis workflows.")
    elif auth_ctx.role == "org_admin":
        st.caption("Role access: Organisation admins can run team workflows and batch pipelines.")
    else:
        st.caption("Role access: Admin users can manage platform settings and all workflows.")
    st.divider()

    # ── Credits ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sidebar-note">
          <div style="font-weight:600; margin-bottom:.2rem;">👤 Developed by Simon Mufara</div>
          Bioinformatics · Machine Learning · Cancer Genomics<br>
          <a href="https://github.com/Simon-Mufara/variant-filtering-dashboard" style="text-decoration:none;">
            🔗 GitHub Repository
          </a><br>
          Built with Streamlit · Plotly · Ensembl VEP · gnomAD · SnpEff · ACMG<br>
          ⚕️ Research use only — not for clinical diagnosis
        </div>
        """,
        unsafe_allow_html=True,
    )

workspace_bits = [
    st.session_state.get("workspace_org", "").strip(),
    st.session_state.get("workspace_team", "").strip(),
    st.session_state.get("workspace_project", "").strip(),
]
workspace_label = " / ".join([v for v in workspace_bits if v])


EX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example.vcf")
EX_ANNOTATED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example_annotated.vcf")
EX_MAF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example.maf")
_UPLOAD_TYPES = supported_extensions()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — SINGLE VCF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

if mode == "🔬 Single VCF":

    with st.sidebar:
        with st.expander(f"{_UI_ICONS['data']} Data Input", expanded=True):
            vcf_file = st.file_uploader(
                "Upload variant file",
                type=_UPLOAD_TYPES,
                help="Accepts: VCF, VCF.GZ, MAF (TCGA), TSV, CSV variant tables"
            )
            st.caption("Supported: VCF · VCF.GZ · MAF · TSV · CSV")
            url_input = st.text_input(
                "Or load from URL (HTTPS/FTP)",
                placeholder="https://…/example.vcf.gz",
                help="Paste a direct URL to a VCF or VCF.GZ file"
            )
            st.divider()
            ex_choice = st.radio(
                "Or use a built-in example",
                ["Plain VCF", "Annotated VCF (SnpEff + ClinVar)", "MAF (TCGA cancer)", "None"],
                index=0,
                help="Pre-loaded examples for testing each tab"
            )

        _ex_map = {
            "Plain VCF": EX_PATH,
            "Annotated VCF (SnpEff + ClinVar)": EX_ANNOTATED_PATH,
            "MAF (TCGA cancer)": EX_MAF_PATH,
        }
        _ex_path = _ex_map.get(ex_choice)
        use_example = (ex_choice != "None") and not vcf_file
        df_raw = _safe_load(vcf_file, use_example, _ex_path or EX_PATH, "variant file")
        if df_raw is None and url_input and not vcf_file:
            try:
                df_raw = _cached_load_url(url_input)
            except Exception as _url_exc:
                st.error(f"❌ Failed to load from URL: {_url_exc}")
                st.stop()
        if df_raw is None or df_raw.empty or "chrom" not in df_raw.columns:
            st.title("🧬 Variant Analysis Suite")
            st.info("Upload a variant file (VCF, MAF, TSV/CSV) or choose a built-in example to begin.")
            st.stop()

        with st.expander(f"{_UI_ICONS['filter']} Variant Filters", expanded=True):
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

        with st.expander(f"{_UI_ICONS['annotation']} Annotations", expanded=False):
            # Auto-enable local parsers when annotated example is active
            _is_annotated = (not vcf_file) and ex_choice in (
                "Annotated VCF (SnpEff + ClinVar)", "MAF (TCGA cancer)")
            genome_build = st.radio(
                "Genome Build", ["GRCh38 (hg38)", "GRCh37 (hg19)"],
                horizontal=True,
                help="Used for VEP and gnomAD queries"
            )
            do_ensembl  = st.checkbox("Gene names (Ensembl)", help="Queries Ensembl REST API")
            do_vep      = st.checkbox("VEP consequences (first 100)", help="SIFT, PolyPhen, HGVS")
            do_gnomad   = st.checkbox("gnomAD population AF (first 50)")
            do_scores   = st.checkbox("Predictor scores (CADD, SpliceAI, REVEL, AlphaMissense)",
                                      value=_is_annotated)
            do_acmg     = st.checkbox("ACMG-lite classification",
                                      value=_is_annotated)
            do_priority = st.checkbox("Variant prioritization score",
                                      value=_is_annotated)
            if _is_annotated:
                st.success("✅ Predictor scores & prioritization auto-enabled for annotated example")
            st.caption("⚠️ API annotations (Ensembl/VEP/gnomAD) require internet; slow for large VCFs")

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
            df = _cached_annotate_vep(df, genome_build=genome_build)
    if do_gnomad and not df.empty:
        with st.spinner("Querying gnomAD (first 50 variants)…"):
            df = _cached_annotate_gnomad(df, genome_build=genome_build)
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
    if workspace_label:
        sub_parts.append(f"🏢 Workspace: **{workspace_label}**")
    st.caption("  ·  ".join(sub_parts))
    st.info(
        "🧭 Navigate left-to-right through tabs: start in **Overview**, "
        "inspect **Statistics**, then export from **Data Table** or **Report**."
    )

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
        "📈 Overview", "📉 Distributions", "🧭 Genome Browser", "👥 Multi-Sample",
        "🎯 Prioritize", "🧬 Gene Panel", "🔎 VEP", "🧪 SnpEff", "🩺 ClinVar",
        "🧬 ACMG", "📊 Statistics", "🧠 Predictors", "🗂️ Data Table", "📝 Report",
    ]
    tabs = st.tabs(tab_names)

    # ── 0: Overview ───────────────────────────────────────────────────────────
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(chromosome_plot(df), width="stretch", key="chrom_plot")
        c2.plotly_chart(variant_type_plot(df), width="stretch", key="variant_type_plot")
        c3, c4 = st.columns(2)
        c3.plotly_chart(quality_distribution(df), width="stretch", key="quality_dist")
        c4.plotly_chart(tstv_plot(df), width="stretch", key="tstv_plot")
        # SV summary
        if "variant_type" in df.columns and (df["variant_type"] == "SV").any():
            st.divider()
            st.markdown('<div class="section-header">🔷 Structural Variant Summary</div>', unsafe_allow_html=True)
            sv_df = df[df["variant_type"] == "SV"].copy()
            c1, c2 = st.columns(2)
            with c1:
                svtype_counts = sv_df["svtype"].value_counts().reset_index() if "svtype" in sv_df.columns else pd.DataFrame()
                if not svtype_counts.empty:
                    svtype_counts.columns = ["SVTYPE", "Count"]
                    st.markdown(f"**{len(sv_df)} SVs detected**")
                    st.dataframe(svtype_counts, width="stretch")
            with c2:
                chrom_counts = sv_df["chrom"].value_counts().reset_index()
                chrom_counts.columns = ["Chromosome", "SV Count"]
                fig_sv = px.bar(chrom_counts, x="Chromosome", y="SV Count", title="SVs per Chromosome")
                st.plotly_chart(fig_sv, width="stretch", key="sv_chrom_plot")
            if "svlen" in sv_df.columns and sv_df["svlen"].abs().gt(0).any():
                sv_df["sv_size"] = sv_df["svlen"].abs()
                fig_size = px.histogram(sv_df[sv_df["sv_size"] > 0], x="sv_size", nbins=30,
                                        title="SV Size Distribution (bp)", log_x=True)
                st.plotly_chart(fig_size, width="stretch", key="sv_size_plot")

    # ── 1: Distributions ──────────────────────────────────────────────────────
    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.plotly_chart(depth_distribution(df), width="stretch", key="depth_dist")
        if "af" in df.columns and df["af"].notna().any():
            c2.plotly_chart(af_scatter(df), width="stretch", key="af_scatter")
        else:
            c2.info("AF data not available in this VCF.")

    # ── 2: Genome Browser ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header">Positional Variant Track</div>', unsafe_allow_html=True)
        if not df.empty and "chrom" in df.columns:
            chrom_sel = st.selectbox("Chromosome", sorted(df["chrom"].unique().tolist(), key=_chrom_sort_key))
            st.plotly_chart(positional_track(df, chrom_sel), width="stretch", key="positional_track")
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
                st.plotly_chart(fig, width="stretch", key=f"genotype_dist_{col}")
        else:
            st.info("No sample genotype columns found in this VCF.")

    # ── 4: Prioritization ─────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header">🎯 Variant Prioritization</div>', unsafe_allow_html=True)
        if "priority_score" not in df.columns:
            st.info(
                "**Variant prioritization score** ranks variants 0–100 using an additive model:\n\n"
                "| Component | Max pts | Source |\n"
                "|---|---|---|\n"
                "| ACMG classification | 40 | ACMG-lite (sidebar) |\n"
                "| gnomAD AF rarity | 20 | gnomAD API (sidebar) |\n"
                "| SnpEff/VEP impact | 15 | ANN= or vep_impact column |\n"
                "| ClinVar significance | 15 | CLNSIG INFO field |\n"
                "| Predictor scores | 10 | CADD, REVEL, AlphaMissense |\n"
                "| QUAL score | 10 | VCF QUAL field |\n\n"
                "**To activate:** tick **Variant prioritization score** in the sidebar.\n\n"
                "💡 For the richest scores, also enable **Predictor scores** and "
                "**ACMG-lite classification** — or select the "
                "**Annotated VCF (SnpEff + ClinVar)** example which activates all three automatically."
            )
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
                st.plotly_chart(fig, width="stretch", key="priority_tier_dist")

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
                st.plotly_chart(fig, width="stretch", key="panel_gene_counts")
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
                        st.plotly_chart(fig, width="stretch", key="vep_impact_dist")

                if "vep_symbol" in df.columns:
                    top_genes = df["vep_symbol"][df["vep_symbol"] != ""].value_counts().head(15).reset_index()
                    top_genes.columns = ["Gene","Count"]
                    if not top_genes.empty:
                        fig2 = px.bar(top_genes, x="Gene", y="Count", title="Top Affected Genes (VEP)")
                        st.plotly_chart(fig2, width="stretch", key="vep_top_genes")

                st.download_button("⬇️ Download VEP Annotations (CSV)",
                                   vep_df.to_csv(index=False).encode(),
                                   "vep_annotations.csv", "text/csv")

    # ── 7: SnpEff ─────────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown('<div class="section-header">🧪 SnpEff Functional Annotation</div>', unsafe_allow_html=True)
        ann_df = parse_snpeff(df)
        if ann_df.empty:
            st.info(
                "No SnpEff **ANN=** field found in this VCF.\n\n"
                "**Quick fix:** In the sidebar, switch the example to "
                "**Annotated VCF (SnpEff + ClinVar)** — it loads immediately with 15 cancer "
                "variants annotated at HIGH/MODERATE/LOW impact.\n\n"
                "**For your own VCF**, pre-annotate with SnpEff:\n"
                "```\nsnpEff ann GRCh38.86 input.vcf > annotated.vcf\n```"
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                imp = impact_summary(ann_df)
                fig = px.bar(imp, x="Impact", y="Count", color="Impact",
                             color_discrete_map=IMPACT_COLORS, title="Variants by Impact Level")
                st.plotly_chart(fig, width="stretch", key="snpeff_impact_dist")
            with c2:
                genes = top_affected_genes(ann_df)
                fig2  = px.bar(genes.head(15), x="Gene", y="Count", color="High Impact",
                               title="Most Frequently Affected Genes")
                st.plotly_chart(fig2, width="stretch", key="snpeff_top_genes")
            st.dataframe(ann_df, width="stretch", height=380)
            st.download_button("⬇️ Download SnpEff Annotations (CSV)",
                               ann_df.to_csv(index=False).encode(),
                               "snpeff_annotations.csv", "text/csv")

    # ── 8: ClinVar ────────────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown('<div class="section-header">🏥 ClinVar Clinical Significance</div>', unsafe_allow_html=True)
        clin_df = clinvar_significance(df)
        if clin_df.empty or clin_df["ClinVar Significance"].eq("Unknown").all():
            st.info(
                "No ClinVar **CLNSIG=** field found in this VCF.\n\n"
                "**Quick fix:** In the sidebar, switch the example to "
                "**Annotated VCF (SnpEff + ClinVar)** — it includes ClinVar "
                "classifications for 15 cancer genes (TP53, BRCA1, KRAS, etc.).\n\n"
                "**For your own VCF**, annotate with ClinVar:\n"
                "```\nbcftools annotate \\\n"
                "  -a clinvar_20240101.vcf.gz \\\n"
                "  -c INFO/CLNSIG,INFO/CLNDN,INFO/CLNREVSTAT \\\n"
                "  input.vcf > clinvar_annotated.vcf\n```\n"
                "Download ClinVar VCF from: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
            )
        else:
            sig_counts = clin_df["ClinVar Significance"].value_counts().reset_index()
            sig_counts.columns = ["Significance","Count"]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(sig_counts, names="Significance", values="Count",
                                   title="ClinVar Significance Distribution"),
                            width="stretch", key="clinvar_sig_pie")
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
                            width="stretch", key="acmg_class_pie")
            c2.plotly_chart(px.bar(acmg_counts, x="Classification", y="Count",
                                   color="Classification", color_discrete_map=COLOR_MAP,
                                   title="ACMG Classification Counts"),
                            width="stretch", key="acmg_class_bar")
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
        qc_tabs = st.tabs(["QC Overview", "Allele Balance", "Variant Density", "Per-Sample"])
        with qc_tabs[0]:
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
                                width="stretch", key="depth_per_chrom")
                st.dataframe(dpc, width="stretch")
        with qc_tabs[1]:
            ab_df = allele_balance_stats(df)
            if ab_df.empty:
                st.info("No AD= allele depth fields found in INFO. Allele balance requires AD=ref,alt in INFO.")
            else:
                fig_ab = px.histogram(ab_df.dropna(subset=["allele_balance"]),
                                      x="allele_balance", nbins=40,
                                      title="Allele Balance Distribution",
                                      labels={"allele_balance": "Allele Balance (alt/(ref+alt))"})
                fig_ab.add_vline(x=0.5, line_dash="dash", line_color="red",
                                 annotation_text="Expected het (0.5)")
                st.plotly_chart(fig_ab, width="stretch", key="allele_balance_dist")
                st.dataframe(ab_df, width="stretch")
        with qc_tabs[2]:
            dens_df = variant_density(df)
            if dens_df.empty:
                st.info("No positional data for density calculation.")
            else:
                fig_dens = px.density_heatmap(dens_df, x="bin", y="chrom", z="count",
                                              title="Variant Density (10 Mb bins)",
                                              labels={"bin": "Genomic Position (Mb)", "chrom": "Chromosome", "count": "Variants"})
                st.plotly_chart(fig_dens, width="stretch", key="variant_density")
                st.dataframe(dens_df, width="stretch")
        with qc_tabs[3]:
            miss_df = missingness_per_sample(df)
            if miss_df.empty:
                st.info("No sample genotype columns found.")
            else:
                fig_miss = px.bar(miss_df, x="sample", y="missingness_pct",
                                  title="Per-Sample Missingness (%)",
                                  labels={"missingness_pct": "Missingness %", "sample": "Sample"})
                st.plotly_chart(fig_miss, width="stretch", key="sample_missingness")
                st.dataframe(miss_df, width="stretch")

    # ── 11: Predictor Scores ──────────────────────────────────────────────────
    with tabs[11]:
        st.markdown('<div class="section-header">📋 Pathogenicity Predictor Scores</div>', unsafe_allow_html=True)
        score_cols = [c for c in ["cadd_phred","revel_score","spliceai_max_delta",
                                  "alphamissense_score","alphamissense_class"] if c in df.columns]
        if not score_cols:
            st.info(
                "**Predictor scores** (CADD, SpliceAI, REVEL, AlphaMissense) are parsed "
                "from the VCF INFO field.\n\n"
                "**To activate:**\n"
                "1. Tick **Predictor scores** in the sidebar → Annotations\n"
                "2. Use an annotated VCF — select **Annotated VCF (SnpEff + ClinVar)** "
                "from the example selector, or upload your own VCF annotated with:\n"
                "```\n"
                "# Annotate with dbNSFP (REVEL, AlphaMissense, CADD):\n"
                "bcftools annotate -a dbNSFP4.4a_grch38.gz \\\n"
                "  -c CHROM,POS,REF,ALT,CADD_PHRED,REVEL,AM_PATHOGENICITY input.vcf\n\n"
                "# Annotate with SpliceAI:\n"
                "spliceai -I input.vcf -O annotated.vcf -R genome.fa -A grch38\n"
                "```"
            )
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
                    st.plotly_chart(fig, width="stretch", key=f"predictor_dist_{col}")

    # ── 12: Data Table ────────────────────────────────────────────────────────
    with tabs[12]:
        st.markdown('<div class="section-header">📋 Filtered Variants</div>', unsafe_allow_html=True)
        display_df = df.drop(columns=["info_raw"], errors="ignore")
        st.dataframe(display_df, width="stretch", height=440)
        c1, c2, c3 = st.columns(3)
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
        try:
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine='openpyxl') as writer:
                display_df.to_excel(writer, sheet_name='Filtered Variants', index=False)
                if do_priority and 'priority_score' in df.columns:
                    df[['chrom', 'position', 'ref', 'alt', 'priority_score', 'priority_tier']].to_excel(
                        writer, sheet_name='Prioritized', index=False)
            c3.download_button(
                "⬇️ Download XLSX (multi-sheet)", xlsx_buf.getvalue(),
                "variants.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception:
            c3.info("XLSX export requires openpyxl (check requirements.txt)")

    # ── 13: Report ────────────────────────────────────────────────────────────
    with tabs[13]:
        st.markdown('<div class="section-header">📄 Export Reports</div>', unsafe_allow_html=True)
        fname_clean = (vcf_file.name if vcf_file else "example.vcf").replace(" ","_")
        case_id = st.session_state.get("workspace_case_id", "").strip().replace(" ", "_")
        report_prefix = f"{case_id}_{fname_clean}" if case_id else fname_clean
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**HTML Report**\nSelf-contained, shareable with colleagues")
            if st.button("🔄 Generate HTML Report", type="primary"):
                with st.spinner("Building HTML report…"):
                    html_bytes = generate_report(df_raw, df, stats, filename=report_prefix)
                st.download_button("⬇️ Download HTML Report", html_bytes,
                                   report_prefix.replace(".vcf","") + "_report.html", "text/html")
        with c2:
            st.markdown("**PDF Report**\nProfessional format for clinical/lab use")
            if not pdf_available():
                st.warning("Install `fpdf2` to enable PDF export: `pip install fpdf2`")
            elif st.button("🔄 Generate PDF Report"):
                with st.spinner("Building PDF report…"):
                    pdf_bytes = generate_pdf(df_raw, df, stats, filename=report_prefix)
                if pdf_bytes:
                    st.download_button("⬇️ Download PDF Report", pdf_bytes,
                                       report_prefix.replace(".vcf","") + "_report.pdf", "application/pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — MULTI-VCF COMPARE (up to 10 files)
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "⚖️ Multi-VCF Compare":

    with st.sidebar:
        with st.expander(f"{_UI_ICONS['data']} Upload VCF Files", expanded=True):
            n_vcfs = st.slider("Number of VCFs to compare", 2, 10, 2)
            uploaded = []
            for i in range(n_vcfs):
                f = st.file_uploader(f"VCF {chr(65+i)}", type=_UPLOAD_TYPES, key=f"cmp_{i}")
                uploaded.append(f)
            use_demo = st.checkbox("Use example VCF for all slots", value=True)

    st.title("⚖️ Multi-VCF Comparison")
    st.info("🧭 Tip: Check the heatmap first, then open detailed pairwise tabs for variant-level review.")

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
    st.plotly_chart(heatmap_fig, width="stretch", key="concordance_heatmap")

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
        plot_cols[i].plotly_chart(variant_type_plot(df_i), width="stretch", key=f"variant_type_plot_{i}")
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
                            width="stretch", key="concordance_by_type")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — TRIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "👨‍👩‍👧 Trio Analysis":

    with st.sidebar:
        with st.expander(f"{_UI_ICONS['data']} Upload Trio VCFs", expanded=True):
            f_proband = st.file_uploader("👶 Proband (affected)", type=_UPLOAD_TYPES, key="trio_prob")
            f_mother  = st.file_uploader("👩 Mother",              type=_UPLOAD_TYPES, key="trio_mom")
            f_father  = st.file_uploader("👨 Father",              type=_UPLOAD_TYPES, key="trio_dad")
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
        with st.expander(f"{_UI_ICONS['data']} Upload Paired VCFs", expanded=True):
            f_tumor  = st.file_uploader("🔬 Tumor VCF",  type=_UPLOAD_TYPES, key="som_tumor")
            f_normal = st.file_uploader("✅ Normal VCF", type=_UPLOAD_TYPES, key="som_normal")
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
            c1.plotly_chart(variant_type_plot(somatic), width="stretch", key="somatic_variant_type_plot")
            c2.plotly_chart(chromosome_plot(somatic), width="stretch", key="somatic_chrom_plot")
            # Clonal architecture VAF plot
            if "af" in somatic.columns and somatic["af"].notna().any():
                st.divider()
                st.markdown('<div class="section-header">🧬 Clonal Architecture</div>', unsafe_allow_html=True)
                vaf_df = somatic[somatic["af"].notna()].copy()
                vaf_df["clonal_tier"] = vaf_df["af"].apply(
                    lambda v: "Clonal (VAF ≥ 0.4)" if v >= 0.4 else (
                        "Subclonal (0.1–0.4)" if v >= 0.1 else "Rare (< 0.1)"
                    )
                )
                tier_colors = {
                    "Clonal (VAF ≥ 0.4)": "#dc2626",
                    "Subclonal (0.1–0.4)": "#ea580c",
                    "Rare (< 0.1)": "#16a34a",
                }
                tier_counts = vaf_df["clonal_tier"].value_counts().reset_index()
                tier_counts.columns = ["Tier", "Count"]
                ca1, ca2 = st.columns(2)
                with ca1:
                    fig_pie = px.pie(tier_counts, names="Tier", values="Count",
                                     color="Tier", color_discrete_map=tier_colors,
                                     title="Clonal Composition")
                    st.plotly_chart(fig_pie, width="stretch", key="clonal_composition_pie")
                with ca2:
                    fig_vaf = px.histogram(vaf_df, x="af", color="clonal_tier",
                                           color_discrete_map=tier_colors,
                                           nbins=40, title="VAF Distribution by Clonal Tier",
                                           labels={"af": "Variant Allele Frequency"})
                    st.plotly_chart(fig_vaf, width="stretch", key="vaf_distribution")
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
# MODE 5 — ADMIN CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "🛠️ Admin Console":
    st.title("🛠️ Admin Console")
    st.caption("Platform administration for organisations, teams, and individual users.")

    if auth_ctx.role != "admin":
        st.error("Only platform admins can access this section.")
        st.stop()

    orgs = list_organizations()
    teams = list_teams()
    users = list_users()
    active_users = [u for u in users if u.get("is_active")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Organisations", len(orgs))
    c2.metric("Teams", len(teams))
    c3.metric("Users", len(users))
    c4.metric("Active Users", len(active_users))

    st.divider()
    tabs = st.tabs(["👥 Users & Roles", "🏢 Workspace Governance", "⚙️ Platform Settings", "📌 Admin Notes"])

    with tabs[0]:
        st.markdown("### Create User Account")

        org_options = {"None": None}
        for org in orgs:
            org_options[f"{org['name']} (#{org['id']})"] = org["id"]

        team_options = {"None": None}
        for team in teams:
            team_options[f"{team['organization_name']} / {team['name']} (#{team['id']})"] = team["id"]

        with st.form("create_user_form"):
            new_username = st.text_input("Username", placeholder="e.g. team.lead")
            new_full_name = st.text_input("Full Name", placeholder="e.g. Grace Mufara")
            new_password = st.text_input("Password", type="password", help="Minimum 8 characters")
            new_role = st.selectbox(
                "Role",
                ["individual", "team_member", "org_admin", "admin"],
                help="Role controls analysis mode access",
            )
            new_org_label = st.selectbox("Organisation", list(org_options.keys()))
            new_team_label = st.selectbox("Team", list(team_options.keys()))
            create_user_submitted = st.form_submit_button("Create User", type="primary")

        if create_user_submitted:
            try:
                org_id = org_options.get(new_org_label)
                team_id = team_options.get(new_team_label)
                create_user_account(
                    username=new_username,
                    full_name=new_full_name,
                    password=new_password,
                    role=new_role,
                    organization_id=org_id,
                    team_id=team_id,
                )
                st.success(f"Created user '{new_username}'.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to create user: {exc}")

        st.divider()
        st.markdown("### Existing Users")
        users_df = pd.DataFrame(users)
        if users_df.empty:
            st.info("No users found.")
        else:
            users_df = users_df.rename(
                columns={
                    "username": "Username",
                    "full_name": "Full Name",
                    "role": "Role",
                    "organization_name": "Organisation",
                    "team_name": "Team",
                    "is_active": "Active",
                    "created_at": "Created",
                }
            )
            st.dataframe(users_df[[c for c in ["id", "Username", "Full Name", "Role", "Organisation", "Team", "Active", "Created"] if c in users_df.columns]], width="stretch")

            user_labels = {
                f"{u['username']} ({u['role']})": u["id"]
                for u in users
                if u["id"] != auth_ctx.user_id
            }
            if user_labels:
                selected_label = st.selectbox("Select user for account status change", list(user_labels.keys()))
                selected_user_id = user_labels[selected_label]
                col_a, col_b = st.columns(2)
                if col_a.button("Deactivate User", type="secondary"):
                    set_user_active(selected_user_id, False)
                    st.success("User deactivated.")
                    st.rerun()
                if col_b.button("Reactivate User", type="secondary"):
                    set_user_active(selected_user_id, True)
                    st.success("User reactivated.")
                    st.rerun()

    with tabs[1]:
        st.markdown("### Create Organisation")
        with st.form("create_org_form"):
            org_name = st.text_input("Organisation Name", placeholder="e.g. School of Health Sciences")
            org_submit = st.form_submit_button("Create Organisation", type="primary")
        if org_submit:
            try:
                create_organization(org_name)
                st.success(f"Organisation '{org_name}' created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to create organisation: {exc}")

        st.divider()
        st.markdown("### Create Team")
        org_lookup = {f"{o['name']} (#{o['id']})": o["id"] for o in orgs}
        if org_lookup:
            with st.form("create_team_form"):
                selected_org_label = st.selectbox("Parent Organisation", list(org_lookup.keys()))
                team_name = st.text_input("Team Name", placeholder="e.g. Precision Oncology Unit")
                team_submit = st.form_submit_button("Create Team", type="primary")
            if team_submit:
                try:
                    create_team(org_lookup[selected_org_label], team_name)
                    st.success(f"Team '{team_name}' created.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to create team: {exc}")
        else:
            st.info("Create at least one organisation before adding teams.")

        st.divider()
        st.markdown("### Current Tenant Structure")
        if orgs:
            org_df = pd.DataFrame(orgs)
            st.dataframe(org_df.rename(columns={"id": "Org ID", "name": "Organisation", "created_at": "Created"}), width="stretch")
        else:
            st.info("No organisations configured.")

        if teams:
            team_df = pd.DataFrame(teams)
            team_df = team_df.rename(
                columns={
                    "id": "Team ID",
                    "name": "Team",
                    "organization_name": "Organisation",
                    "created_at": "Created",
                }
            )
            st.dataframe(team_df[[c for c in ["Team ID", "Team", "Organisation", "Created"] if c in team_df.columns]], width="stretch")
        else:
            st.info("No teams configured.")

    with tabs[2]:
        st.markdown("### Security & Session")
        st.markdown(
            "- Passwords are stored as PBKDF2 hashes in the app user database.\n"
            "- Session timeout is controlled by `AUTH_SESSION_TIMEOUT_MIN`.\n"
            "- Use different user accounts for each team member for accountability."
        )

        st.markdown("### Recommendation")
        st.success(
            "For production: use the Admin Console to provision per-user accounts, "
            "set a strong session timeout, and rotate admin credentials on a schedule."
        )

    with tabs[3]:
        st.text_area(
            "Operational Notes",
            key="admin_operational_notes",
            placeholder="Track approvals, data releases, SOP updates, and onboarding notes.",
            height=180,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 6 — BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

elif mode == "📦 Batch Pipeline":

    with st.sidebar:
        with st.expander("⚙️ Pipeline Settings", expanded=True):
            min_qual_batch = st.slider("Min Quality", 0, 100, DEFAULT_MIN_QUAL, key="batch_qual")
            min_dp_batch   = st.slider("Min Depth", 0, 500, DEFAULT_MIN_DP,    key="batch_dp")
            do_acmg_batch  = st.checkbox("ACMG-lite classification", key="batch_acmg")
            do_scores_batch = st.checkbox("Parse predictor scores", key="batch_scores")
            do_vep_batch = st.checkbox("Generate VEP command plan", key="batch_vep_plan")
            include_fastq_pipeline = st.checkbox("Show FASTQ variant-calling workflow", key="batch_fastq_plan")

    st.title("📦 Batch Pipeline")
    st.info("Upload multiple VCF files. All will be filtered with the same settings "
            "and merged into a single annotated CSV download.")

    st.markdown('<div class="section-header">🧰 Automation Assistant</div>', unsafe_allow_html=True)
    automation_tabs = st.tabs(["🧪 Predictor/VEP Automation", "🧬 FASTQ → VCF Workflow"])

    with automation_tabs[0]:
        st.markdown(
            "Generate ready-to-run command templates for annotation. "
            "These commands can be copied to your Linux terminal."
        )
        genome_build_cmd = st.selectbox(
            "Reference build for command templates",
            ["grch38", "grch37"],
            key="automation_genome_build",
        )
        input_vcf_name = st.text_input(
            "Input VCF filename",
            value="input.vcf.gz",
            key="automation_input_vcf",
        )
        output_prefix = st.text_input(
            "Output prefix",
            value="annotated_output",
            key="automation_output_prefix",
        )

        predictor_cmd = (
            "bcftools annotate "
            f"-a dbNSFP4.4a_{genome_build_cmd}.gz "
            "-c CHROM,POS,REF,ALT,CADD_PHRED,REVEL,AM_PATHOGENICITY "
            f"{input_vcf_name} -Oz -o {output_prefix}.dbnsfp.vcf.gz"
        )
        spliceai_cmd = (
            "spliceai "
            f"-I {output_prefix}.dbnsfp.vcf.gz "
            f"-O {output_prefix}.predictors.vcf.gz "
            f"-R genome_{genome_build_cmd}.fa -A {genome_build_cmd}"
        )
        vep_cmd = (
            "vep "
            f"-i {output_prefix}.predictors.vcf.gz "
            f"-o {output_prefix}.vep.vcf "
            "--vcf --everything --offline --cache "
            f"--assembly {'GRCh38' if genome_build_cmd == 'grch38' else 'GRCh37'}"
        )

        st.code(predictor_cmd, language="bash")
        st.code(spliceai_cmd, language="bash")
        st.code(vep_cmd, language="bash")

        st.download_button(
            "⬇️ Download annotation command script",
            (
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                f"{predictor_cmd}\n{spliceai_cmd}\n{vep_cmd}\n"
            ).encode(),
            "run_annotation_pipeline.sh",
            "text/x-shellscript",
        )

        st.markdown("**Tool availability check on this host**")
        _render_tool_status(["bcftools", "spliceai", "vep"])

    with automation_tabs[1]:
        st.markdown(
            "If users only have FASTQ files, this workflow generates a best-practice command plan "
            "from raw reads to a VCF suitable for this app."
        )
        sample_id = st.text_input("Sample ID", value="SAMPLE001", key="fastq_sample_id")
        ref_fa = st.text_input("Reference FASTA path", value="genome.fa", key="fastq_ref_fa")
        fastq_r1 = st.text_input("FASTQ R1 path", value="sample_R1.fastq.gz", key="fastq_r1")
        fastq_r2 = st.text_input("FASTQ R2 path", value="sample_R2.fastq.gz", key="fastq_r2")

        fastq_pipeline = f"""#!/usr/bin/env bash
set -euo pipefail

# 1) Align reads
bwa mem -t 8 {ref_fa} {fastq_r1} {fastq_r2} | samtools sort -@ 8 -o {sample_id}.sorted.bam
samtools index {sample_id}.sorted.bam

# 2) Mark duplicates
gatk MarkDuplicates -I {sample_id}.sorted.bam -O {sample_id}.dedup.bam -M {sample_id}.dup_metrics.txt
samtools index {sample_id}.dedup.bam

# 3) Variant calling
bcftools mpileup -f {ref_fa} {sample_id}.dedup.bam | bcftools call -mv -Oz -o {sample_id}.raw.vcf.gz
bcftools index {sample_id}.raw.vcf.gz

# 4) Basic filtering
bcftools filter -e 'QUAL<30 || DP<10' {sample_id}.raw.vcf.gz -Oz -o {sample_id}.filtered.vcf.gz
bcftools index {sample_id}.filtered.vcf.gz

# 5) Optional annotation (predictors + VEP)
bcftools annotate -a dbNSFP4.4a_grch38.gz -c CHROM,POS,REF,ALT,CADD_PHRED,REVEL,AM_PATHOGENICITY {sample_id}.filtered.vcf.gz -Oz -o {sample_id}.dbnsfp.vcf.gz
spliceai -I {sample_id}.dbnsfp.vcf.gz -O {sample_id}.predictors.vcf.gz -R {ref_fa} -A grch38
vep -i {sample_id}.predictors.vcf.gz -o {sample_id}.vep.vcf --vcf --everything --offline --cache --assembly GRCh38
"""
        st.code(fastq_pipeline, language="bash")
        st.download_button(
            "⬇️ Download FASTQ-to-VCF pipeline script",
            fastq_pipeline.encode(),
            f"{sample_id.lower()}_fastq_to_vcf.sh",
            "text/x-shellscript",
        )
        st.markdown("**Tool availability check on this host**")
        _render_tool_status(["bwa", "samtools", "gatk", "bcftools", "spliceai", "vep"])
        st.caption(
            "Tip: run these commands on a compute server/HPC. Then upload the final VCF here."
        )

    batch_files = st.file_uploader("Upload VCF files (up to 20)",
                                   type=_UPLOAD_TYPES,
                                   accept_multiple_files=True,
                                   key="batch_upload")

    if not batch_files:
        st.caption("Upload VCF files above to begin batch processing.")
        if not include_fastq_pipeline and not do_vep_batch:
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
            if do_vep_batch:
                vep_plan = (
                    "# Run VEP for each filtered file in your batch\n"
                    "# Example:\n"
                    "for f in *.filtered.vcf.gz; do\n"
                    "  vep -i \"$f\" -o \"${f%.vcf.gz}.vep.vcf\" --vcf --everything --offline --cache --assembly GRCh38\n"
                    "done\n"
                )
                st.code(vep_plan, language="bash")
                st.download_button(
                    "⬇️ Download batch VEP plan",
                    vep_plan.encode(),
                    "batch_vep_plan.sh",
                    "text/x-shellscript",
                )
        else:
            st.error("No VCFs were processed successfully.")
