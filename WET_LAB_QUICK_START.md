# Wet-Lab Batch Context: Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Enable Full Pipeline Mode

**Single VCF Mode:**
1. Upload your VCF file in the sidebar
2. Under "Sample metadata", check the box: **"📊 Full Pipeline Mode"**
3. A new section appears: **"Wet-Lab Batch Context"**

**Batch Pipeline Mode:**
- Full Pipeline Mode is **enabled by default** for batch analysis

### Step 2: Quick-Fill Metadata

#### For DNA Extraction:
1. Click "Extraction kit Template" dropdown
2. Select your kit (e.g., "DNeasy Blood & Tissue Kit")
3. ✅ Extraction parameters pre-filled automatically
4. Scroll down and enter **QC measurements:**
   - 260/280 ratio
   - 260/230 ratio
   - Qubit concentration (ng/µL)
   - Fragment integrity (%)

#### For Library Prep:
1. Click "Library Kit Template" dropdown
2. Select your kit (e.g., "Illumina TruSeq DNA")
3. Enter **PCR cycles** (typically 12–18)
4. Enter **library concentration** (nM)

#### For Sequencing:
1. Click "Sequencing Platform Template"
2. Select your platform (e.g., "Illumina NovaSeq 6000")
3. Enter **target depth** (30M for WGS, 100M+ for WES)

### Step 3: Check Risk Flags

Go to the **"Risk Assessment"** tab in the Wet-Lab Context panel:

- 🟢 **No flags** = Green light, proceed normally
- ⚠️ **Warning flags** = Review the recommended mitigations
- 🚨 **Critical flags** = Investigate before finalizing results

### Step 4: Interpret Variants

In the **Overview tab** of your analysis:
- A new section appears: **"Wet-Lab Provenance Context"**
- Shows batch ID, dates, and applicable risk flags
- **Batch Confidence Modifier** tells you if quality is reduced

**Example confidence adjustments:**
```
Normal batch:        Variant confidence unchanged
Protein contamination:  Reduce variant confidence by ~15%
Low DNA yield:         Reduce variant confidence by ~40%
Low yield + excess PCR: Reduce variant confidence by ~50%
```

---

## 🎯 Common Scenarios

### Scenario 1: Standard WGS Analysis (30M reads)

**Input:**
- DNeasy Blood & Tissue Kit
- Illumina TruSeq DNA
- NovaSeq 6000, 30M reads

**Expected result:**
✅ No risk flags, high confidence (>80%)

---

### Scenario 2: Low DNA Starting Material

**Input:**
- 8 ng/µL Qubit concentration (low)
- 28 PCR cycles (excess)

**Expected result:**
🚨 **Critical: Low input + excess amplification**
- Confidence modified to ~50–60%
- Recommendation: Re-sample if available; validate orthogonally

**Action:**
1. ❌ Do NOT report low-frequency variants as definitive
2. ✅ Focus on high-DP (>50x), high-QUAL (>100) calls
3. ✅ Validate by PCR or Sanger sequencing

---

### Scenario 3: Protein Contamination Detected

**Input:**
- 260/280 ratio = 1.5 (low, indicates protein)

**Expected result:**
⚠️ **Warning: Protein contamination**
- May see: uneven coverage, library inefficiency
- Recommended mitigation: column cleanup or re-extraction

**Action:**
1. Check coverage uniformity (depth distribution)
2. Look for regions with very low depth
3. Consider increasing coverage to offset efficiency loss

---

### Scenario 4: Batch Consistency Check (50 samples)

**Action:**
1. Go to "Batch QC" tab
2. Review "Batch Consistency Validation" heatmap
3. Look for samples with low consistency scores

**If all samples show ~100% consistency:**
✅ Batch is uniform, safe to combine for analysis

**If some samples show <80% consistency:**
⚠️ Investigate those samples—may have processing variations

---

## 📋 Data Entry Checklist

### Minimum Required
- [ ] Extraction kit name
- [ ] Extraction date
- [ ] Qubit DNA concentration
- [ ] 260/280 and 260/230 ratios
- [ ] Library kit name
- [ ] PCR cycle count
- [ ] Sequencing platform
- [ ] Target depth (reads)

### Optional but Recommended
- [ ] Technician name
- [ ] Sample preservation method
- [ ] Storage temperature
- [ ] Proteinase K lot number
- [ ] Sequencing operator
- [ ] Operator notes (any issues?)

### Advanced (Only if available)
- [ ] Fragment integrity % (Bioanalyzer/TapeStation)
- [ ] Demultiplexing software version
- [ ] Basecalling software version
- [ ] Library concentration (nM)

---

## 🔍 What to Look For in Risk Flags

### 🚨 CRITICAL — Act immediately

| Flag | Cause | Action |
|---|---|---|
| Low DNA Yield | <10 ng/µL | Warn clinician; apply strict filtering |
| DNA Degradation | <80% integrity | Increase filtering stringency; validate findings |
| Low Input + Excess PCR | Yield <10 ng/µL + >20 cycles | Re-sample if possible |

### ⚠️ WARNING — Proceed with caution

| Flag | Cause | Action |
|---|---|---|
| Protein Contamination | 260/280 <1.7 | Check coverage uniformity |
| Salt Contamination | 260/230 <2.0 | May see PCR inhibition; validate high-confidence calls |
| Excess PCR | >25 cycles | Scrutinize rare variants; check for GC bias |

### ℹ️ INFO — Document and note

| Flag | Cause | Action |
|---|---|---|
| Excessive freeze-thaw | >5 cycles | Document; monitor variant distribution |
| Low library concentration | <2 nM | May affect per-sample evenness |

---

## 💾 Exporting Results

### Single VCF Mode
1. Complete your analysis (apply filters, annotations, etc.)
2. Go to **"Data Table"** or **"Report"** tabs
3. Download CSV or PDF
4. **Wet-lab context automatically included** in exports

### Batch Pipeline Mode
1. Upload VCF files
2. Click **"Run Batch Pipeline"**
3. Download:
   - **Combined CSV:** All variants + batch metadata columns
   - **Batch Metadata (JSON):** Full provenance record (archive this!)
   - **VEP plan:** Ready-to-run annotation commands

**Example JSON export:**
```json
{
  "batch_id": "BATCH-2026-05-12-001",
  "batch_date": "2026-05-12",
  "extraction": {
    "extraction_kit": "DNeasy Blood & Tissue Kit",
    "qubit_concentration_ng_ul": 50.2,
    "qc": {
      "nanodrop_260_280": 1.82,
      "nanodrop_260_230": 2.15,
      "qc_pass_fail": "Pass"
    }
  },
  "library_prep": {
    "kit_name": "Illumina TruSeq DNA",
    "pcr_cycles": 15
  },
  "sequencing": {
    "platform": "Illumina",
    "instrument_model": "NovaSeq 6000"
  },
  "batch_confidence_modifier": 0.95
}
```

---

## ❓ FAQ

**Q: What if I don't have all the QC measurements?**
A: Enter what you have. The system validates available fields and flags missing critical ones.

**Q: Can I modify metadata after processing?**
A: Yes! Modify in the "Wet-Lab Batch Context" panel and re-run analysis. Changes update automatically.

**Q: What does "batch_confidence_modifier" mean?**
A: It's a 0–1 multiplier applied to variant confidence scores. 
- 1.0 = no risk, confidence unchanged
- 0.9 = small risk, confidence reduced 10%
- 0.6 = major risk, confidence reduced 40%

**Q: How do I validate findings from a flagged batch?**
A: Use orthogonal methods:
1. PCR + Sanger sequencing (confirmed de novo variants)
2. Repeat extraction from same sample
3. MLPA panel (copy number CNVs)
4. qPCR (for VAF validation)

**Q: Can I use this for clinical reporting?**
A: Yes, with caution:
1. ✅ Document wet-lab metadata in your report
2. ✅ Flag high-risk samples
3. ✅ Apply batch confidence modifiers to reported variants
4. ⚠️ Validate critical findings orthogonally before finalizing
5. ⚠️ Consult your lab's SOP for wet-lab QC thresholds

**Q: Does this replace standard QC procedures?**
A: No! This system **complements** your existing QC:
- Multiqc for sequencing QC ✓
- Samtools flagstat for alignment QC ✓
- This system for **lab chemistry context** ← new

---

## 🔗 Related Resources

- Full documentation: [WET_LAB_PROVENANCE_GUIDE.md](WET_LAB_PROVENANCE_GUIDE.md)
- API reference: See docstrings in `utils/wet_lab_provenance.py`
- GitHub issues: https://github.com/Simon-Mufara/variant-filtering-dashboard

---

## 🎓 Learn More

### Key Concepts

**Nanodrop 260/280 Ratio**
- Measures DNA purity relative to protein
- 1.7–2.0 = pure DNA
- <1.7 = protein contamination (Proteinase K incomplete digestion)
- >2.0 = RNA contamination

**Nanodrop 260/230 Ratio**
- Measures salt/phenol contamination
- >2.0 = good purification
- <2.0 = salt/phenol residue (PCR inhibition risk)

**Qubit Concentration**
- Fluorescence-based DNA quantitation (more accurate than Nanodrop)
- Expected: ≥10 ng/µL for library prep
- <10 ng/µL = low yield → allelic dropout risk

**Fragment Integrity**
- Measured by Bioanalyzer (RNA) or TapeStation (DNA)
- ≥80% = high molecular weight DNA (intact)
- <80% = degraded (fragmented) DNA → mapping artifacts

**PCR Cycles**
- Library amplification step
- 12–18 cycles = normal (minimal amplification bias)
- >25 cycles = risk of:
  - Over-representation of high-GC regions
  - False variant inflation
  - Clonal artifacts

---

## 📞 Support

Encountering issues? Please file an issue on GitHub:
https://github.com/Simon-Mufara/variant-filtering-dashboard/issues

Include:
1. Your batch metadata (JSON export)
2. Screenshot of the risk flags
3. Expected vs. observed results
4. Steps to reproduce

---

**Last updated:** May 2026
**Version:** 1.0.0
