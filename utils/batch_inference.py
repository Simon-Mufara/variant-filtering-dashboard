"""
Batch Inference Engine

Predicts downstream issues based on wet-lab metadata.
Generates risk flags and confidence modifiers for variant interpretation.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

from utils.wet_lab_provenance import (
    BatchMetadata, QCValidator, ExtractionMetadata, LibraryPrepMetadata
)


class RiskSeverity(Enum):
    """Risk flag severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RiskFlag:
    """Single downstream risk flag inferred from wet-lab metadata."""
    severity: RiskSeverity
    category: str  # "extraction", "library", "sequencing", "amplification", "contamination"
    title: str
    observed_cause: str
    expected_downstream_symptom: str
    interpretation_caution: str
    suggested_mitigation: str
    confidence_modifier: float  # 0.0 to 1.0; applied to variant confidence


class DownstreamInferenceEngine:
    """Infers downstream issues from wet-lab metadata."""
    
    @staticmethod
    def infer_risks(batch: BatchMetadata) -> List[RiskFlag]:
        """Generate all applicable risk flags for a batch."""
        flags = []
        
        # Extraction-based risks
        flags.extend(DownstreamInferenceEngine._infer_extraction_risks(batch.extraction))
        
        # Library prep risks
        flags.extend(DownstreamInferenceEngine._infer_library_risks(batch.library_prep))
        
        # Sequencing risks
        flags.extend(DownstreamInferenceEngine._infer_sequencing_risks(batch.sequencing))
        
        # Cross-module risks
        flags.extend(DownstreamInferenceEngine._infer_combined_risks(batch))
        
        return flags
    
    @staticmethod
    def _infer_extraction_risks(ext: ExtractionMetadata) -> List[RiskFlag]:
        """Infer risks from extraction metadata."""
        flags = []
        qc_flags = QCValidator.check_extraction_qc(ext.qc)
        chem_flags = QCValidator.check_extraction_chemistry(ext)
        
        # Low 260/280 → protein contamination
        if "low_260_280_ratio_protein_contamination" in qc_flags or (
            ext.qc.nanodrop_260_280 and ext.qc.nanodrop_260_280 < 1.7
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="contamination",
                title="Protein Contamination Detected",
                observed_cause=f"260/280 ratio: {ext.qc.nanodrop_260_280} (expected 1.7–2.0)",
                expected_downstream_symptom="Poor library efficiency, reduced sequencing coverage, increased duplicate reads",
                interpretation_caution="Variants may have lower mappability; assess read depth and mapping quality carefully",
                suggested_mitigation="Re-extract or use cleanup columns; consider re-sequencing if coverage severely affected",
                confidence_modifier=0.85
            ))
        
        # Low 260/230 → salt contamination
        if "low_260_230_ratio_salt_contamination" in qc_flags or (
            ext.qc.nanodrop_260_230 and ext.qc.nanodrop_260_230 < 2.0
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="contamination",
                title="Salt/Phenol Contamination Suspected",
                observed_cause=f"260/230 ratio: {ext.qc.nanodrop_260_230} (expected >2.0)",
                expected_downstream_symptom="PCR inhibition, reduced library efficiency, uneven amplification",
                interpretation_caution="May cause variable coverage across regions; suspect low-confidence variants",
                suggested_mitigation="Additional ethanol wash or re-precipitation; consider quantification by fluorometry (Qubit)",
                confidence_modifier=0.80
            ))
        
        # Low DNA yield
        if "low_dna_yield" in qc_flags or (
            ext.qc.qubit_concentration_ng_ul and ext.qc.qubit_concentration_ng_ul < 10.0
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.CRITICAL,
                category="extraction",
                title="Low DNA Yield",
                observed_cause=f"Qubit concentration: {ext.qc.qubit_concentration_ng_ul} ng/µL (threshold: 10 ng/µL)",
                expected_downstream_symptom="Low-complexity library, uneven amplification, potential allelic dropout, non-random sampling",
                interpretation_caution="Risk of false negatives (missed variants) and biased variant calls; genotype confidence unreliable",
                suggested_mitigation="Re-extract if sample available; document in variant calls; apply stricter filtering",
                confidence_modifier=0.60
            ))
        
        # Degraded DNA
        if "degraded_dna_low_integrity" in qc_flags or (
            ext.qc.fragment_integrity_percent and ext.qc.fragment_integrity_percent < 80.0
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.CRITICAL,
                category="extraction",
                title="DNA Degradation Detected",
                observed_cause=f"Fragment integrity: {ext.qc.fragment_integrity_percent}% (expected >80%)",
                expected_downstream_symptom="Fragmented mapping artifacts, false variants at read boundaries, increased false positives",
                interpretation_caution="Elevated risk of spurious variants; carefully review breakpoint-spanning reads and paired-end consistency",
                suggested_mitigation="Re-extract if possible; use lenient mapping quality thresholds cautiously; prioritize high-coverage variants",
                confidence_modifier=0.65
            ))
        
        # Excessive freeze-thaw
        if "excessive_freeze_thaw_dna_degradation" in chem_flags or ext.freeze_thaw_cycles > 5:
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="extraction",
                title="Excessive Freeze-Thaw Cycles",
                observed_cause=f"Freeze-thaw cycles: {ext.freeze_thaw_cycles} (recommended ≤2)",
                expected_downstream_symptom="DNA degradation, reduced complexity, biased variant representation",
                interpretation_caution="Monitor for non-random variant distribution and unexpected allele frequency skewing",
                suggested_mitigation="Minimize future freeze-thaw; document in metadata; validate key findings",
                confidence_modifier=0.80
            ))
        
        # Low proteinase K
        if "low_proteinase_k_concentration" in chem_flags or (
            ext.protein_digestion.concentration_ug_ml and ext.protein_digestion.concentration_ug_ml < 0.1
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="contamination",
                title="Suboptimal Proteinase K Digestion",
                observed_cause=f"Proteinase K: {ext.protein_digestion.concentration_ug_ml} µg/mL (recommended ≥0.1 µg/mL)",
                expected_downstream_symptom="Residual protein, poor library efficiency, potential inhibition",
                interpretation_caution="Check for protein contamination in QC; may see uneven coverage or reduced mappability",
                suggested_mitigation="Increase proteinase K concentration in future extractions; consider column re-binding",
                confidence_modifier=0.85
            ))
        
        return flags
    
    @staticmethod
    def _infer_library_risks(lib: LibraryPrepMetadata) -> List[RiskFlag]:
        """Infer risks from library prep metadata."""
        flags = []
        qc_flags = QCValidator.check_library_prep_qc(lib)
        
        # Excess PCR cycles → amplification bias
        if "excess_pcr_cycles_amplification_bias" in qc_flags or (
            lib.pcr_cycles and lib.pcr_cycles > 25
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="amplification",
                title="Excess PCR Amplification Cycles",
                observed_cause=f"PCR cycles: {lib.pcr_cycles} (recommended ≤15–18 for WGS)",
                expected_downstream_symptom="Amplification bias, inflated VAF variance, false variants in high-GC regions, artificial CNV signals",
                interpretation_caution="Over-represented variants likely near sequencing adapters; apply bias correction; scrutinize rare variants",
                suggested_mitigation="Reduce PCR cycles; use highly processable libraries; normalize by GC content",
                confidence_modifier=0.70
            ))
        
        # Low library concentration
        if "low_library_concentration" in qc_flags or (
            lib.library_concentration_nm and lib.library_concentration_nm < 2.0
        ):
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="library",
                title="Low Library Concentration",
                observed_cause=f"Library concentration: {lib.library_concentration_nm} nM (expected ≥2 nM)",
                expected_downstream_symptom="Low sequencing yield, variable coverage, incomplete target representation",
                interpretation_caution="May result in uneven per-sample depth; validate by sequencing metrics post-run",
                suggested_mitigation="Re-amplify or re-ligate; verify library size distribution; consider individual sample optimization",
                confidence_modifier=0.80
            ))
        
        return flags
    
    @staticmethod
    def _infer_sequencing_risks(seq) -> List[RiskFlag]:
        """Infer risks from sequencing metadata."""
        flags = []
        
        # Sequencing depth too low (if specified)
        if seq.sequencing_depth_target_million and seq.sequencing_depth_target_million < 20:
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="sequencing",
                title="Low Target Sequencing Depth",
                observed_cause=f"Target depth: {seq.sequencing_depth_target_million}M reads (WGS recommended ≥30M)",
                expected_downstream_symptom="Insufficient coverage for variant detection, high false negative rate, unreliable genotypes at low-frequency",
                interpretation_caution="Increase confidence thresholds; may miss heterozygous variants; prioritize high-confidence calls",
                suggested_mitigation="Increase sequencing depth or re-sequence; prioritize high-interest regions",
                confidence_modifier=0.75
            ))
        
        return flags
    
    @staticmethod
    def _infer_combined_risks(batch: BatchMetadata) -> List[RiskFlag]:
        """Infer risks from combination of extraction, library, and sequencing metadata."""
        flags = []
        
        # Low DNA yield + high PCR cycles
        if (batch.extraction.qc.qubit_concentration_ng_ul and 
            batch.extraction.qc.qubit_concentration_ng_ul < 10.0 and
            batch.library_prep.pcr_cycles and 
            batch.library_prep.pcr_cycles > 20):
            flags.append(RiskFlag(
                severity=RiskSeverity.CRITICAL,
                category="amplification",
                title="Low Input + Excess Amplification",
                observed_cause=f"Low yield ({batch.extraction.qc.qubit_concentration_ng_ul} ng/µL) + {batch.library_prep.pcr_cycles} PCR cycles",
                expected_downstream_symptom="Severe amplification bias, non-random variant inflation, genotyping errors, clonal artifacts",
                interpretation_caution="High false positive and false negative risk; variants may not reflect true biological signal",
                suggested_mitigation="Critical: re-sample and re-extract if available; apply strict filtering; validate orthogonally",
                confidence_modifier=0.50
            ))
        
        # Batch-level operator notes indicating anomalies
        if batch.operator_notes and any(x in batch.operator_notes.lower() for x in ["problem", "issue", "failed", "anomal", "investigate"]):
            flags.append(RiskFlag(
                severity=RiskSeverity.WARNING,
                category="extraction",
                title="Operator-Flagged Anomaly",
                observed_cause=f"Operator notes: {batch.operator_notes[:100]}",
                expected_downstream_symptom="Undefined; review operator notes for details",
                interpretation_caution="Investigate specific issue noted by operator before interpreting variants",
                suggested_mitigation="Contact operator for clarification; may require re-processing",
                confidence_modifier=0.85
            ))
        
        return flags
    
    @staticmethod
    def compute_batch_confidence_modifier(batch: BatchMetadata) -> float:
        """
        Compute overall confidence modifier for the entire batch (0.0–1.0).
        Lower values indicate higher risk.
        """
        flags = DownstreamInferenceEngine.infer_risks(batch)
        
        if not flags:
            return 1.0
        
        # Aggregate confidence modifiers
        modifiers = [f.confidence_modifier for f in flags]
        
        # Weight by severity
        severity_weights = {
            RiskSeverity.INFO: 0.05,
            RiskSeverity.WARNING: 0.20,
            RiskSeverity.CRITICAL: 0.50
        }
        
        total_weight = 0.0
        weighted_modifier = 1.0
        
        for i, flag in enumerate(flags):
            weight = severity_weights.get(flag.severity, 0.1)
            weighted_modifier *= (modifiers[i] ** weight)
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_modifier = weighted_modifier ** (1.0 / total_weight) if total_weight <= 1.0 else weighted_modifier
        
        return max(0.0, min(1.0, weighted_modifier))
    
    @staticmethod
    def flag_to_markdown(flag: RiskFlag) -> str:
        """Format a RiskFlag as Markdown for Streamlit display."""
        severity_emoji = {
            RiskSeverity.INFO: "ℹ️",
            RiskSeverity.WARNING: "⚠️",
            RiskSeverity.CRITICAL: "🚨",
        }
        emoji = severity_emoji.get(flag.severity, "•")
        
        return f"""
{emoji} **{flag.title}** [{flag.severity.value.upper()}]

**Observed Cause:** {flag.observed_cause}

**Expected Downstream Symptom:** {flag.expected_downstream_symptom}

**Interpretation Caution:** {flag.interpretation_caution}

**Suggested Mitigation:** {flag.suggested_mitigation}

**Confidence Modifier:** {flag.confidence_modifier:.2f}
"""


class VariantConfidenceAssessor:
    """Assesses confidence of individual variants based on wet-lab context."""
    
    @staticmethod
    def assess_variant_confidence(
        variant_quality: float,
        variant_depth: float,
        batch_confidence_modifier: float,
        wet_lab_risk_flags: List[RiskFlag]
    ) -> Dict[str, any]:
        """
        Assess confidence of a single variant accounting for wet-lab context.
        
        Returns:
            Dict with confidence_score (0–100), interpretation_level, and notes.
        """
        # Base confidence from variant metrics (0–100)
        base_confidence = min(100.0, (variant_quality / 60.0) * 50 + (variant_depth / 30.0) * 50)
        
        # Apply batch confidence modifier
        adjusted_confidence = base_confidence * batch_confidence_modifier
        
        # Determine interpretation level
        if adjusted_confidence >= 80:
            interpretation = "High confidence"
            color_class = "high"
        elif adjusted_confidence >= 60:
            interpretation = "Moderate caution"
            color_class = "moderate"
        elif adjusted_confidence >= 40:
            interpretation = "Wet-lab uncertainty detected"
            color_class = "low"
        else:
            interpretation = "Potential technical artifact"
            color_class = "critical"
        
        # Generate notes
        notes = []
        critical_flags = [f for f in wet_lab_risk_flags if f.severity == RiskSeverity.CRITICAL]
        if critical_flags:
            notes.append(f"Critical: {len(critical_flags)} high-risk wet-lab flags detected")
        
        warning_flags = [f for f in wet_lab_risk_flags if f.severity == RiskSeverity.WARNING]
        if warning_flags:
            notes.append(f"Warning: {len(warning_flags)} moderate-risk wet-lab flags detected")
        
        return {
            "base_confidence": base_confidence,
            "adjusted_confidence": adjusted_confidence,
            "interpretation_level": interpretation,
            "color_class": color_class,
            "batch_modifier": batch_confidence_modifier,
            "notes": notes
        }
