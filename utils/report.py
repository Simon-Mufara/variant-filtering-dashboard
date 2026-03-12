"""HTML report generator — produces a self-contained, downloadable report."""
from __future__ import annotations
import datetime
import pandas as pd


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem 3rem; color: #1e293b; background: #f8fafc; }
h1 { color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: .4rem; }
h2 { color: #1e40af; margin-top: 2rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
               gap: 1rem; margin: 1rem 0; }
.metric { background: white; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.05); }
.metric .value { font-size: 2rem; font-weight: 700; color: #3b82f6; }
.metric .label { font-size: .85rem; color: #64748b; margin-top: .2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0;
        font-size: .85rem; background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.05); }
th { background: #1e40af; color: white; padding: .6rem 1rem; text-align: left; }
td { padding: .5rem 1rem; border-bottom: 1px solid #e2e8f0; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f8fafc; }
.tag { display: inline-block; padding: .15rem .5rem; border-radius: 12px;
       font-size: .75rem; font-weight: 600; }
.tag-path  { background: #fee2e2; color: #b91c1c; }
.tag-lpath { background: #fed7aa; color: #c2410c; }
.tag-vus   { background: #fef9c3; color: #854d0e; }
.tag-lben  { background: #d1fae5; color: #065f46; }
.tag-ben   { background: #e0f2fe; color: #075985; }
footer { margin-top: 3rem; color: #94a3b8; font-size: .8rem; border-top: 1px solid #e2e8f0;
         padding-top: 1rem; }
"""

_ACMG_TAG = {
    "Pathogenic": "tag-path",
    "Likely Pathogenic": "tag-lpath",
    "VUS": "tag-vus",
    "Likely Benign": "tag-lben",
    "Benign": "tag-ben",
}


def generate_report(
    df_raw: pd.DataFrame,
    df_filtered: pd.DataFrame,
    stats: dict,
    filename: str = "variants",
) -> bytes:
    """Generate a self-contained HTML report as bytes.

    Args:
        df_raw:      full (unfiltered) variant DataFrame
        df_filtered: filtered variant DataFrame
        stats:       dict from utils.stats.variant_stats()
        filename:    used in the report title

    Returns:
        UTF-8 encoded HTML bytes suitable for st.download_button.
    """
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pass_rate = round(len(df_filtered) / len(df_raw) * 100, 1) if len(df_raw) else 0

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

    metric_html = "\n".join(
        f'<div class="metric"><div class="value">{v}</div><div class="label">{k}</div></div>'
        for k, v in metrics
    )

    # Variant type breakdown table
    type_rows = ""
    if "variant_type" in df_filtered.columns:
        for vt, cnt in df_filtered["variant_type"].value_counts().items():
            pct = round(cnt / len(df_filtered) * 100, 1) if len(df_filtered) else 0
            type_rows += f"<tr><td>{vt}</td><td>{cnt}</td><td>{pct}%</td></tr>"

    # Chromosome table
    chrom_rows = ""
    if "chrom" in df_filtered.columns:
        for ch, cnt in df_filtered["chrom"].value_counts().head(25).items():
            chrom_rows += f"<tr><td>{ch}</td><td>{cnt}</td></tr>"

    # ACMG table (if present)
    acmg_section = ""
    if "acmg_class" in df_filtered.columns:
        acmg_counts = df_filtered["acmg_class"].value_counts()
        rows = ""
        for cls, cnt in acmg_counts.items():
            tag = _ACMG_TAG.get(cls, "tag-vus")
            rows += f'<tr><td><span class="tag {tag}">{cls}</span></td><td>{cnt}</td></tr>'
        acmg_section = f"""
        <h2>🧬 ACMG Classification Summary</h2>
        <table>
          <thead><tr><th>Classification</th><th>Count</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    # Top variants table (first 20 rows, drop info_raw)
    display = df_filtered.drop(columns=["info_raw"], errors="ignore").head(20)
    variant_table = _df_to_html(display)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Variant Report — {filename}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>🧬 Variant Analysis Report</h1>
  <p><strong>File:</strong> {filename} &nbsp;|&nbsp; <strong>Generated:</strong> {ts}</p>

  <h2>📊 Summary Metrics</h2>
  <div class="metric-grid">{metric_html}</div>

  <h2>🔢 Variant Type Breakdown</h2>
  <table>
    <thead><tr><th>Type</th><th>Count</th><th>%</th></tr></thead>
    <tbody>{type_rows}</tbody>
  </table>

  <h2>🗺️ Variants per Chromosome (top 25)</h2>
  <table>
    <thead><tr><th>Chromosome</th><th>Count</th></tr></thead>
    <tbody>{chrom_rows}</tbody>
  </table>

  {acmg_section}

  <h2>📋 Top 20 Filtered Variants</h2>
  {variant_table}

  <footer>
    Generated by <strong>Variant Analysis Suite</strong> &nbsp;•&nbsp; {ts}
  </footer>
</body>
</html>"""

    return html.encode("utf-8")


def _df_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>No variants to display.</em></p>"
    header = "".join(f"<th>{col}</th>" for col in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row.values)
        rows += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
