# Wet-Lab Batch Provenance System

## Overview

The Wet-Lab Batch Provenance System is an advanced module for the Variant Analysis Suite that captures, validates, and leverages upstream laboratory metadata to inform downstream variant analysis quality and interpretation.

**Purpose:** Bridge the gap between wet-lab operations and genomic data analysis by documenting critical upstream chemistry that affects variant quality.

## Key Features

### 1. **Comprehensive Metadata Capture**

Captures structured wet-lab metadata across four critical domains:

#### DNA Extraction Metadata
- **Extraction kit** and custom protocols
- **Sample type** (whole blood, tissue, cell line, etc.)
- **Extraction date** and technician information
- **Storage conditions** and freeze-thaw cycles
- **Lysis parameters:** buffer type, temperature, incubation time
- **Protein digestion:** Proteinase K concentration, enzyme lot numbers
- **Alcohol purification:** type, concentration, wash counts
- **Binding chemistry:** silica column vs. magnetic bead specifications
- **Elution parameters:** buffer volume, repetitions
- **Post-extraction QC:** Nanodrop 260/280, 260/230, Qubit concentration, fragment integrity

#### Library Preparation Metadata
- **Library kit** name and version
- **Input DNA amount**
- **Fragmentation:** method, duration, target/measured fragment size
- **PCR amplification:** polymerase, cycle count, adapter kit
- **Cleanup:** bead ratio, normalization method
- **Library QC:** concentration, pass/fail status

#### Sequencing Metadata
- **Platform and instrument** (Illumina, PacBio, Nanopore, etc.)
- **Flowcell ID** and lane information
- **Read parameters:** length, type (paired-end/single-end)
- **Run date** and target depth
- **Software versions:** basecalling and demultiplexing
- **Sequencing operator**

### 2. **Automated QC Validation**

Real-time flagging of abnormal values:

| QC Parameter | Normal Range | Issue |
|---|---|---|
| **260/280 ratio** | 1.7–2.0 | <1.7 → protein contamination; >2.0 → RNA contamination |
| **260/230 ratio** | >2.0 | <2.0 → salt/phenol contamination |
| **Qubit conc.** | ≥10 ng/µL | <10 ng/µL → low DNA yield, allelic dropout risk |
| **Fragment integrity** | ≥80% | <80% → degraded DNA, mapping artifacts |
| **Proteinase K** | ≥0.1 µg/mL | <0.1 → incomplete digestion, inhibition |
| **PCR cycles** | 12–18 | >25 → amplification bias, false variants |

### 3. **Downstream Inference Engine**

Predicts quality issues and generates risk flags:

**Example 1: Low 260/230 ratio (salt contamination)**
- **Observed Cause:** 260/230 = 1.2 (expected >2.0)
- **Expected Downstream Symptom:** PCR inhibition, uneven amplification
- **Interpretation Caution:** Check for variable coverage; suspect low-confidence variants
- **Suggested Mitigation:** Additional ethanol wash or Qubit-based quantification

**Example 2: Excess PCR cycles (amplification bias)**
- **Observed Cause:** 28 PCR cycles (recommended ≤15–18)
- **Expected Downstream Symptom:** Artificial variant inflation, GC bias, false variants
- **Interpretation Caution:** Over-represented variants likely near adapters; scrutinize rare variants
- **Suggested Mitigation:** Reduce cycles; apply GC normalization

**Example 3: Low DNA yield + High PCR (CRITICAL)**
- **Observed Cause:** 8 ng/µL DNA + 26 PCR cycles
- **Expected Downstream Symptom:** Clonal artifacts, allelic dropout, genotyping errors
- **Interpretation Caution:** HIGH false positive and false negative risk
- **Suggested Mitigation:** Re-sample; apply strict filtering; validate orthogonally

### 4. **Batch Consistency Validation**

Identifies suspicious variations across samples in a 50–500 sample batch:

- Flags per-sample overrides that deviate from batch defaults
- Generates uniformity heatmap showing per-sample consistency scores
- Highlights critical inconsistencies (e.g., samples sequenced on different platforms)
- Supports rapid investigation of anomalies

### 5. **Sample Provenance Timeline**

Visualizes full lifecycle of each sample:

```
Collection → Storage → Extraction → QC → Library Prep → Sequencing → VCF → Analysis
```

Each stage displays:
- Event type and timestamp
- Pass/Warning/Fail status
- Key metadata (kit name, concentration, operator)

### 6. **Variant Confidence Modifier**

Adjusts variant interpretation confidence based on batch context:

```
Adjusted Confidence = Base Confidence × Batch Modifier
```

**Confidence Levels:**
- **High Confidence (≥80%):** High-quality variant, low wet-lab risk
- **Moderate Caution (60–80%):** Generally reliable; note wet-lab context
- **Wet-Lab Uncertainty (40–60%):** Multiple risk factors; proceed cautiously
- **Potential Technical Artifact (<40%):** Investigate upstream chemistry before interpretation

### 7. **Predefined Templates**

Quick templates for common kits and platforms:

**Extraction Kits:**
- DNeasy Blood & Tissue Kit
- CTAB Extraction
- Phenol-Chloroform

**Library Kits:**
- Illumina TruSeq DNA
- NEBNext Ultra II
- KAPA Hyper Prep

**Sequencing Platforms:**
- Illumina NovaSeq 6000
- Illumina NextSeq 550
- PacBio Sequel II

## How to Use

### Single VCF Mode

1. **Enable Full Pipeline Mode:**
   - In the sidebar under "Sample metadata", check "📊 Full Pipeline Mode"

2. **Capture Wet-Lab Context:**
   - A collapsible "Wet-Lab Batch Context" section appears
   - Fill in DNA Extraction, Library Prep, and Sequencing metadata
   - Use predefined templates to auto-populate common kits

3. **Review Risk Assessment:**
   - "Risk Assessment" tab shows flagged issues with severity (Critical/Warning)
   - Each flag includes observed cause, downstream impact, and mitigation

4. **Interpret with Wet-Lab Awareness:**
   - In the **Overview** tab, "Wet-Lab Provenance Context" card shows:
     - Batch ID and dates
     - Risk flags applicable to all variants
     - Batch confidence modifier
   - Use this context when reviewing variants

### Batch Pipeline Mode

1. **Enable Wet-Lab Tracking:**
   - Checkbox "✅ Enable Full Pipeline Mode (Wet-Lab Provenance)" is pre-checked
   - Recommended for comprehensive batch analysis

2. **Define Batch Parameters:**
   - Fill "Batch ID", "Extraction Date", "Extraction Kit", etc.
   - Quick-fill using predefined templates

3. **Upload VCF Files:**
   - Upload 2–500 VCF files from a single batch

4. **Run Batch Pipeline:**
   - All VCFs processed with same filters
   - Each variant gets batch metadata columns:
     - `batch_id`
     - `batch_date`
     - `extraction_kit`
     - `library_kit`
     - `sequencing_platform`
     - `batch_confidence_modifier`

5. **Export Deliverables:**
   - **Combined CSV:** All variants with batch context
   - **Batch Metadata (JSON):** Full wet-lab provenance for archival
   - **VEP command plan:** Ready-to-run annotation script

## Data Schema

### Core Classes

```python
# Main batch container
BatchMetadata(
    batch_id: str                                    # Unique ID
    batch_name: str                                  # Display name
    batch_date: Optional[str]                        # ISO format: YYYY-MM-DD
    extraction: ExtractionMetadata                   # DNA extraction details
    library_prep: LibraryPrepMetadata               # Library preparation
    sequencing: SequencingMetadata                  # Sequencing run info
    sample_overrides: Dict[str, Dict[str, Any]]    # Per-sample customizations
    operator_notes: str                              # Free-text notes
    qc_flags: List[str]                            # Automated QC flags
    qc_pass_overall: str                            # "Pass", "Fail", "Conditional"
)

# Post-extraction QC
PostExtractionQC(
    nanodrop_260_280: Optional[float]              # Protein contamination indicator
    nanodrop_260_230: Optional[float]              # Salt contamination indicator
    qubit_concentration_ng_ul: Optional[float]      # DNA yield
    fragment_integrity_percent: Optional[float]     # Bioanalyzer/TapeStation
    qc_pass_fail: str                              # "Pass", "Fail", "Warning"
)

# Risk flag (exported to variants)
RiskFlag(
    severity: RiskSeverity                         # CRITICAL, WARNING, INFO
    category: str                                   # extraction, library, sequencing...
    title: str                                      # Display title
    observed_cause: str                             # What was measured
    expected_downstream_symptom: str               # Expected impact
    interpretation_caution: str                     # Interpretation guidance
    suggested_mitigation: str                       # Remediation steps
    confidence_modifier: float                      # 0.0–1.0 applied to variant confidence
)
```

### Serialization

```python
# Save to JSON
batch_json = batch_to_json(batch_metadata)

# Load from JSON
batch_metadata = batch_from_json(batch_json)
```

## Integration with Variant Interpretation

### Variant Confidence Assessment

When viewing individual variants, the system computes:

```
Adjusted Confidence = Base Confidence × Batch Confidence Modifier
```

Example:
- Variant QUAL = 60, Depth = 40 → Base confidence ≈ 75%
- Batch has protein contamination + excess PCR → Modifier = 0.75
- **Adjusted confidence = 75% × 0.75 = 56% (Moderate Caution)**

### Confidence Color Coding

| Confidence | Color | Interpretation |
|---|---|---|
| ≥80% | 🟢 Green | High confidence — proceed normally |
| 60–80% | 🟡 Yellow | Moderate caution — note wet-lab context |
| 40–60% | 🟠 Orange | Wet-lab uncertainty detected — validate |
| <40% | 🔴 Red | Potential technical artifact — investigate |

## Best Practices

### When Entering Metadata

1. **Use predefined templates** when available to avoid errors
2. **Provide extraction kit name exactly** — enables automatic validation rules
3. **Fill in enzyme lot numbers** — supports batch traceability
4. **Document anomalies in operator notes** — informs downstream analysis
5. **Update QC measurements immediately after testing** — reduces memory/transcription errors

### When Interpreting Results

1. **Check batch confidence modifier** before accepting rare variants
2. **Review critical risk flags** (marked 🚨) — may explain unexpected findings
3. **Cross-check sample timeline** if suspecting contamination or degradation
4. **Validate findings** from flagged batches using orthogonal approaches (PCR, Sanger, etc.)
5. **Document wet-lab context in variant reports** — essential for clinical validation

### For Large Batches (50–500 samples)

1. **Use batch consistency validator** to identify outlier samples
2. **Apply per-sample overrides** sparingly (flag suspicious deviations)
3. **Group samples by processing date/kit** when practical
4. **Export batch metadata with results** for future reference
5. **Archive full JSON metadata** alongside variant calls

## Computational Performance

- **Metadata validation:** <100 ms per batch
- **Risk inference:** <50 ms (all rules evaluated)
- **Timeline generation:** <10 ms per sample
- **Batch consistency check:** <200 ms for 500 samples
- **Serialization:** <5 ms per batch (JSON)

## Troubleshooting

### Issue: "QC Flags: low_260_280_ratio_protein_contamination"

**Diagnosis:** 260/280 ratio < 1.7

**Solutions:**
1. Check for RNA contamination (RIN >7 expected)
2. Repeat Nanodrop measurement
3. Clean cuvette with 70% ethanol
4. If persistent: re-extract or use column cleanup

### Issue: Batch confidence modifier very low (<0.5)

**Diagnosis:** Multiple critical risk flags detected

**Solutions:**
1. Review each flag in "Risk Assessment" tab
2. Prioritize **re-sequencing** if DNA yield critically low
3. For variants from affected batch: apply stricter DP/QUAL thresholds
4. Validate key findings orthogonally (PCR/Sanger)

### Issue: Per-sample override not saving

**Diagnosis:** Sample ID format mismatch or UI session timeout

**Solutions:**
1. Verify sample ID matches exactly (case-sensitive)
2. Re-enter override and save explicitly
3. Check browser console for JavaScript errors

## Advanced Usage

### Custom QC Reference Ranges

To modify default QC thresholds, edit [utils/wet_lab_provenance.py](utils/wet_lab_provenance.py):

```python
class QCValidator:
    QC_REFERENCE = {
        "nanodrop_260_280_min": 1.7,  # Adjust as needed
        "nanodrop_260_280_max": 2.0,
        "nanodrop_260_230_min": 2.0,
        # ... etc
    }
```

### Custom Risk Rules

Add new inference rules in [utils/batch_inference.py](utils/batch_inference.py):

```python
@staticmethod
def _infer_custom_risks(batch: BatchMetadata) -> List[RiskFlag]:
    flags = []
    if some_condition(batch):
        flags.append(RiskFlag(
            severity=RiskSeverity.WARNING,
            category="custom",
            title="Your custom flag",
            # ... etc
        ))
    return flags
```

## API Reference

### Core Functions (utils/wet_lab_provenance.py)

```python
# Batch creation
create_batch_from_template(batch_id, extraction_template, library_template, sequencing_template)

# QC validation
QCValidator.check_extraction_qc(qc: PostExtractionQC) -> List[str]
QCValidator.check_library_prep_qc(lib: LibraryPrepMetadata) -> List[str]

# Batch consistency
BatchConsistencyValidator.validate_batch_uniformity(batch, sample_ids) -> Dict
BatchConsistencyValidator.generate_consistency_heatmap_data(batch, sample_ids) -> Dict

# Sample timeline
SampleProvenanceTimeline.build_timeline(batch, sample_id) -> List[TimelineEvent]

# Serialization
batch_to_json(batch: BatchMetadata) -> str
batch_from_json(json_str: str) -> BatchMetadata
```

### Inference Engine (utils/batch_inference.py)

```python
# Risk prediction
DownstreamInferenceEngine.infer_risks(batch: BatchMetadata) -> List[RiskFlag]
DownstreamInferenceEngine.compute_batch_confidence_modifier(batch) -> float

# Variant assessment
VariantConfidenceAssessor.assess_variant_confidence(
    variant_quality, variant_depth, batch_modifier, risk_flags
) -> Dict
```

### UI Components (utils/streamlit_wet_lab_ui.py)

```python
# Session management
initialize_batch_session_state()
get_batch_from_session() -> BatchMetadata
save_batch_to_session(batch)

# Display functions
render_collapsible_batch_context()
render_batch_overview(batch)
render_extraction_input(batch) -> BatchMetadata
render_library_prep_input(batch) -> BatchMetadata
render_sequencing_input(batch) -> BatchMetadata
render_wet_lab_risk_flags(batch)
render_batch_consistency(batch, sample_ids)
render_sample_timeline(batch, sample_id)
render_variant_wet_lab_context(quality, depth, batch)
```

## Compliance & Reproducibility

The Wet-Lab Batch Provenance System supports:

- ✅ **FAIR data principles:** Findable (batch ID), Accessible (JSON export), Interoperable (standard schema), Reusable (full metadata)
- ✅ **Good Laboratory Practice (GLP):** Documented protocols, QC traceability, operator attribution
- ✅ **Clinical validation workflows:** Confidence modifiers help triage variants for orthogonal validation
- ✅ **Batch effect correction:** Enables statistical correction when combining batches across projects

## References

- Nanodrop specification: ThermoFisher Scientific ND-1000 Manual
- Qubit quantitation: ThermoFisher Qubit dsDNA HS Assay Kit
- Fragment analysis: Agilent Bioanalyzer 2100 User Guide
- ACMG guidelines: "Standards and guidelines for the interpretation of sequence variants" (2015)
- Good Laboratory Practice: ISO 17025:2017 General requirements for the competence of testing and calibration laboratories

## License

This module is part of the Variant Analysis Suite. See [LICENSE](../LICENSE) for details.

## Support

For issues, questions, or contributions, visit:
https://github.com/Simon-Mufara/variant-filtering-dashboard/issues
