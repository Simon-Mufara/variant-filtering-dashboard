"""PDF report generator using fpdf2 (pure Python, no system dependencies).

Produces a clean two-column PDF with:
  - Title page + generated timestamp
  - Summary metrics table
  - Variant type breakdown
  - Per-chromosome counts
  - ACMG classification (if available)
  - Priority tier distribution (if available)
  - Top 30 variants table
"""
from __future__ import annotations
import io
import datetime
import pandas as pd

try:
    from fpdf import FPDF  # fpdf2
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False


def pdf_available() -> bool:
    return _FPDF_AVAILABLE


def generate_pdf(
    df_raw: pd.DataFrame,
    df_filtered: pd.DataFrame,
    stats: dict,
    filename: str = "variants",
) -> bytes | None:
    """Generate a PDF report. Returns bytes or None if fpdf2 not installed."""
    if not _FPDF_AVAILABLE:
        return None

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pass_rate = round(len(df_filtered) / len(df_raw) * 100, 1) if len(df_raw) else 0

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_fill_color(30, 64, 175)   # #1e40af
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "Variant Analysis Report", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"File: {filename}   |   Generated: {ts}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # ── Summary metrics ───────────────────────────────────────────────────────
    _section(pdf, "Summary Metrics")
    metrics = [
        ("Total Variants", len(df_raw)),
        ("Passing Filter", len(df_filtered)),
        ("Pass Rate", f"{pass_rate}%"),
        ("Ts/Tv Ratio", stats.get("tstv_ratio", "—")),
        ("Mean Depth", stats.get("mean_depth", "—")),
        ("Mean QUAL", stats.get("mean_qual", "—")),
        ("SNPs", stats.get("snp_count", "—")),
        ("INDELs", stats.get("indel_count", "—")),
    ]
    _two_col_metrics(pdf, metrics)

    # ── Variant type breakdown ─────────────────────────────────────────────────
    pdf.ln(4)
    _section(pdf, "Variant Type Breakdown")
    if "variant_type" in df_filtered.columns:
        rows = []
        for vt, cnt in df_filtered["variant_type"].value_counts().items():
            pct = round(cnt / len(df_filtered) * 100, 1) if len(df_filtered) else 0
            rows.append([str(vt), str(cnt), f"{pct}%"])
        _table(pdf, ["Type", "Count", "%"], rows)

    # ── Per-chromosome ────────────────────────────────────────────────────────
    pdf.ln(4)
    _section(pdf, "Variants per Chromosome (top 20)")
    if "chrom" in df_filtered.columns:
        rows = [[str(ch), str(cnt)]
                for ch, cnt in df_filtered["chrom"].value_counts().head(20).items()]
        _table(pdf, ["Chromosome", "Count"], rows)

    # ── ACMG ──────────────────────────────────────────────────────────────────
    if "acmg_class" in df_filtered.columns:
        pdf.ln(4)
        _section(pdf, "ACMG Classification Summary")
        rows = [[str(cls), str(cnt)]
                for cls, cnt in df_filtered["acmg_class"].value_counts().items()]
        _table(pdf, ["Classification", "Count"], rows)

    # ── Priority tiers ────────────────────────────────────────────────────────
    if "priority_tier" in df_filtered.columns:
        pdf.ln(4)
        _section(pdf, "Priority Tier Distribution")
        rows = [[str(t), str(c)]
                for t, c in df_filtered["priority_tier"].value_counts().items()]
        _table(pdf, ["Tier", "Count"], rows)

    # ── Top variants ──────────────────────────────────────────────────────────
    pdf.add_page()
    _section(pdf, "Top 30 Filtered Variants")
    cols = ["chrom", "position", "ref", "alt", "variant_type", "quality", "depth"]
    for extra in ("acmg_class", "priority_score", "vep_symbol"):
        if extra in df_filtered.columns:
            cols.append(extra)
    display = df_filtered[[c for c in cols if c in df_filtered.columns]].head(30)
    _table(pdf, list(display.columns), display.astype(str).values.tolist(), font_size=7)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5,
                   "DISCLAIMER: This report is for research purposes only. "
                   "ACMG-lite classification is a triage tool and is NOT suitable for clinical decision-making. "
                   "Confirm findings with a certified clinical geneticist.")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ── helpers ───────────────────────────────────────────────────────────────────

def _section(pdf: "FPDF", title: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 64, 175)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 180, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(30, 30, 30)


def _two_col_metrics(pdf: "FPDF", metrics: list) -> None:
    pdf.set_font("Helvetica", "", 10)
    col_w = 90
    for i, (label, value) in enumerate(metrics):
        if i % 2 == 0 and i > 0:
            pdf.ln(7)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w, 7, f"{label}:", border="B")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(col_w, 7, str(value), border="B")
    pdf.ln(9)


def _table(pdf: "FPDF", headers: list, rows: list, font_size: int = 9) -> None:
    n_cols = len(headers)
    col_w = 190 / n_cols

    # Header row
    pdf.set_fill_color(59, 130, 246)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", font_size)
    for h in headers:
        pdf.cell(col_w, 6, str(h)[:20], border=1, fill=True)
    pdf.ln()

    # Data rows
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", font_size)
    for i, row in enumerate(rows):
        fill = i % 2 == 1
        if fill:
            pdf.set_fill_color(240, 245, 255)
        for cell in row:
            pdf.cell(col_w, 5, str(cell)[:22], border=1, fill=fill)
        pdf.ln()
    pdf.ln(3)
