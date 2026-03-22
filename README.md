# 🧬 Variant Analysis Suite

[![Live App](https://img.shields.io/badge/Live%20App-Open%20on%20Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://simon-variants.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Genomic%20Variant%20Analysis-0A9396)](https://simon-variants.streamlit.app)

Clinical and research-focused web application for end-to-end genomic variant analysis, built to convert raw VCF data into interpretable genomic and translational insights.

## Live Platform

- Application: https://simon-variants.streamlit.app
- Repository: Simon-Mufara/variant-filtering-dashboard
- Current status: deployed and active

## What Problem This App Solves

Genomic variant files are information-dense but difficult to interpret consistently across teams. Most workflows require command-line expertise, custom scripts, and manual annotation lookups — creating bottlenecks for researchers, students, and clinicians who need answers, not code.

Variant Analysis Suite solves this by providing one guided platform that standardizes the full workflow from VCF upload to clinical interpretation and report generation.

## What The App Does

The platform delivers a complete 10-stage analysis experience:

1. Upload and validation of VCF, VCF.GZ, MAF, TSV, and CSV files.
2. Variant overview with counts, type breakdown, and quality metrics.
3. Distribution plots for depth, allele frequency, and transition/transversion ratio.
4. Genome browser with per-chromosome positional tracks and IGV integration.
5. Multi-sample genotype comparison across individuals.
6. Functional annotation via SnpEff impact levels and affected genes.
7. Clinical significance lookup using ClinVar annotations.
8. ACMG-lite pathogenicity classification.
9. Comprehensive QC statistics including het/hom ratio and missingness.
10. Report generation as a self-contained HTML file for sharing and collaboration.

## Key Capabilities

- Handles single and multi-sample VCF files in a guided interface.
- Compares two VCFs side-by-side with concordance metrics.
- Overlays population frequency data from gnomAD for the first 50 variants.
- Performs gene name resolution via the Ensembl REST API.
- Applies ACMG-lite rules-based pathogenicity scoring with sidebar toggles.
- Generates downloadable CSV and VCF exports from any filtered view.
- Produces structured HTML reports for meetings and collaborations.

## Outputs Available To Users

- Variant counts, type distributions, and quality summaries.
- Per-chromosome positional tracks and genome browser view.
- Multi-sample genotype tables and distribution charts.
- SnpEff functional impact tables with top affected genes.
- ClinVar clinical significance classifications.
- ACMG-lite pathogenicity calls with supporting criteria.
- Comprehensive QC report with Ts/Tv, depth, and missingness statistics.
- Filtered variant table (CSV or VCF download).
- Self-contained HTML report.

## Who This Is For

- Faculty research groups and genomics labs.
- Clinical and translational research teams.
- Postgraduate students and trainees in genomics.
- Bioinformatics-supported wet-lab and sequencing projects.

## Analysis Modes

| Mode | Description |
|------|-------------|
| Single VCF | Full 10-tab analysis of one sample or multi-sample file |
| Multi-VCF Compare | Side-by-side concordance and distribution comparison of two VCFs |
| Trio Analysis | Family-based variant interpretation |
| Somatic (Tumor/Normal) | Paired tumor-normal comparison |
| Batch Pipeline | Process multiple files in sequence |

## User Guideline (Quick Start)

Use this workflow to keep analysis consistent and clinically sensible:

1. **Load data**
   - Upload `VCF/VCF.GZ/MAF/TSV/CSV`, or start with a built-in example.

2. **Apply baseline quality filters**
   - Set `QUAL`, `DP`, chromosome selection, AF range, and PASS-only if needed.

3. **Add annotation layers**
   - Enable VEP, gnomAD, predictor parsing, and ACMG-lite for richer interpretation.

4. **Interpret in sequence**
   - Start with `Overview` → move to `Prioritize` / `ClinVar` / `ACMG` → validate with `Statistics`.

5. **Export and communicate**
   - Download filtered tables and generate HTML/PDF reports for review meetings.

## Visual Analysis Flow

`Upload` → `Filter` → `Annotate` → `Interpret` → `QC Check` → `Export Report`

Recommended tab journey in **Single VCF** mode:

- `📈 Overview` (initial signal)
- `🎯 Prioritize` + `🩺 ClinVar` + `🧬 ACMG` (biological/clinical triage)
- `📊 Statistics` (quality confidence)
- `🗂️ Data Table` + `📝 Report` (final outputs)

## Partnership Value

- Standardized and reproducible variant analysis practice across projects.
- Faster transition from raw sequencing data to interpretable results.
- Better communication of findings to multidisciplinary audiences.
- Training-friendly environment for onboarding students and new team members.

## Technology

- Streamlit
- Plotly
- Ensembl REST API
- gnomAD GraphQL API
- SnpEff annotation parsing
- ClinVar clinical significance integration
- ACMG-lite classification engine

## Clinical Disclaimer

ACMG-lite is a simplified triage tool only. It does not replace clinical-grade tools such as VarSome, InterVar, or a certified clinical geneticist. Do not use for clinical decision-making.

## License

MIT License. See [LICENSE](LICENSE).
