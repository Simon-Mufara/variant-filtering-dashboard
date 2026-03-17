# 🧬 Variant Analysis Suite

[![CI](https://github.com/Simon-Mufara/variant-filtering-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Simon-Mufara/variant-filtering-dashboard/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)



An industry-grade, interactive dashboard for end-to-end genomic variant analysis — from raw VCF to clinical interpretation.

---

## ✨ Features

### 🔬 Single VCF Analysis (10 tabs)
| Tab | Description |
|-----|-------------|
| 📊 Overview | Variant counts, type breakdown, quality & Ts/Tv plots |
| 📈 Distributions | Depth histogram, allele-frequency scatter |
| 🗺️ Genome Browser | Per-chromosome positional track + IGV Web App launcher |
| 👥 Multi-Sample | Per-sample genotype table and distribution charts |
| 🧪 SnpEff | Functional annotation from `ANN=` INFO field — impact levels, top genes |
| 🏥 ClinVar | CLNSIG/CLNDN clinical significance from annotated VCFs |
| 🧬 ACMG | ACMG-lite rules-based pathogenicity classification (PVS1/PS1/PM2/BA1…) |
| 📉 Statistics | Comprehensive QC — Ts/Tv, het/hom, missingness, depth per chromosome |
| 📋 Data Table | Download filtered variants as CSV or VCF |
| 📄 Report | Generate a self-contained HTML report to share with colleagues |

### ⚖️ Compare Two VCFs
- Shared / unique variant counts with concordance %
- Per-type concordance breakdown
- Side-by-side quality and type distributions

### 🧰 Annotation Overlays (sidebar toggles)
- **Ensembl** — gene name lookup via REST API
- **gnomAD** — population allele frequencies (first 50 variants)
- **ACMG-lite** — rules-based pathogenicity scoring

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Simon-Mufara/variant-filtering-dashboard.git
cd variant-filtering-dashboard

# 2. Create environment
conda create -n variant_dashboard python=3.11
conda activate variant_dashboard

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# 39 tests — vcf_parser, filters, validator, ACMG
```

CI runs automatically on every push via GitHub Actions.

---

## 📁 Project Structure

```
variant-filtering-dashboard/
├── app.py                       # Streamlit entry point (10 tabs + Compare mode)
├── config.py                    # Default thresholds and constants
├── utils/
│   ├── vcf_parser.py            # Pure-Python VCF parser (multi-sample, gzip)
│   ├── filters.py               # NaN-safe filtering (quality, depth, AF, chrom, PASS)
│   ├── plots.py                 # Plotly visualisations + Ensembl gene annotation
│   ├── compare.py               # VCF-vs-VCF comparison & concordance
│   ├── snpeff.py                # SnpEff ANN field parser — impact + gene tables
│   ├── stats.py                 # Comprehensive QC stats + ClinVar parsing
│   ├── acmg.py                  # ACMG-lite variant classification (PVS1/PS1/PM2/BA1…)
│   ├── gnomad.py                # gnomAD GraphQL API population frequency lookup
│   ├── report.py                # Self-contained HTML report generator
│   ├── validator.py             # VCF file validation before parsing
│   └── logger.py                # Centralised logging
├── data/
│   └── example.vcf              # Multi-sample VCF for demo/testing
├── tests/
│   ├── test_vcf_parser.py       # VCF parser unit tests
│   ├── test_filters.py          # Filter unit tests
│   └── test_validator_acmg.py   # Validator + ACMG unit tests
├── .github/
│   └── workflows/ci.yml         # GitHub Actions CI (pytest + flake8)
├── .streamlit/
│   └── config.toml              # 2 GB upload limit
├── requirements.txt
└── runtime.txt                  # python-3.11
```

---

## ⚠️ Clinical Disclaimer

ACMG-lite is a simplified triage tool only. It does **not** replace clinical-grade tools such as [VarSome](https://varsome.com), [InterVar](http://www.intervar.org/), or a certified clinical geneticist. Do not use for clinical decision-making.

---

## 📄 License

MIT

