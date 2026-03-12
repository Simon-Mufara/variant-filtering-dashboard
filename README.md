# Variant Filtering Dashboard

Interactive dashboard for exploring and filtering genomic variants from VCF files.

## Features

- Upload VCF files (single-sample and multi-sample)
- Filter variants by quality, depth, type, chromosome, and allele frequency
- Interactive visualizations: quality, depth, AF, Ts/Tv ratio, variant types, per-chromosome counts
- Genome browser — positional variant track per chromosome
- Gene annotation via Ensembl REST API
- Per-sample genotype breakdown for multi-sample VCFs
- Export filtered variants as CSV

## Installation

```bash
conda create -n variant_dashboard python=3.10
conda activate variant_dashboard
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Run Tests

```bash
pytest tests/
```

## Project Structure

```
variant-filtering-dashboard/
├── app.py                      # Streamlit application entry point
├── config.py                   # Default thresholds and constants
├── utils/
│   ├── vcf_parser.py           # VCF parsing via pysam (single + multi-sample)
│   ├── filters.py              # Filter logic (quality, depth, type, AF, chrom)
│   └── plots.py                # Plotly visualisations + Ensembl annotation
├── data/
│   └── example.vcf             # Multi-sample VCF for testing
├── tests/
│   ├── test_vcf_parser.py      # Parser unit tests
│   └── test_filters.py         # Filter unit tests
└── requirements.txt
```
