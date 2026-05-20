"""
Wet-Lab Provenance Module

Captures upstream laboratory metadata affecting downstream variant analysis quality.
Provides structured schemas for DNA extraction, library prep, sequencing, and QC metadata.
Integrates with batch processing and enables downstream risk inference.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import json


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LysisMetadata:
    """DNA extraction lysis details."""
    buffer_type: str = "Unknown"  # e.g., "Tris-HCl", "Guanidinium thiocyanate"
    buffer_version: str = ""
    incubation_temperature_c: Optional[float] = None
    incubation_duration_min: Optional[float] = None
    notes: str = ""


@dataclass
class ProteinDigestionMetadata:
    """Proteinase K digestion parameters."""
    concentration_ug_ml: Optional[float] = None
    incubation_temperature_c: Optional[float] = None
    incubation_duration_min: Optional[float] = None
    enzyme_lot_number: str = ""
    notes: str = ""


@dataclass
class AlcoholPurificationMetadata:
    """Alcohol purification step details."""
    alcohol_type: str = "Ethanol"  # "Ethanol" or "Isopropanol"
    concentration_percent: Optional[float] = 100.0
    wash_count: int = 2
    notes: str = ""


@dataclass
class BindingChemistry:
    """DNA binding method in extraction."""
    method: str = "Silica column"  # "Silica column", "Magnetic bead", "Other"
    column_type: str = ""
    magnetic_bead_type: str = ""
    custom_notes: str = ""


@dataclass
class ElutionMetadata:
    """Elution parameters post-extraction."""
    buffer_type: str = "TE buffer"
    buffer_volume_ul: Optional[float] = None
    elution_repetitions: int = 1
    final_volume_ul: Optional[float] = None
    notes: str = ""


@dataclass
class PostExtractionQC:
    """Post-extraction quality control metrics."""
    nanodrop_260_280: Optional[float] = None
    nanodrop_260_230: Optional[float] = None
    qubit_concentration_ng_ul: Optional[float] = None
    fragment_integrity_percent: Optional[float] = None
    fragment_integrity_method: str = ""  # "Bioanalyzer", "TapeStation", "Fragment Analyzer"
    qc_pass_fail: str = "Unknown"  # "Pass", "Fail", "Warning", "Unknown"
    qc_notes: str = ""


@dataclass
class ExtractionMetadata:
    """Complete DNA extraction metadata."""
    extraction_kit: str = ""
    custom_extraction_protocol: str = ""
    sample_type: str = "Unknown"  # "Whole blood", "Tissue", "Cell line", etc.
    extraction_date: Optional[str] = None  # ISO format: YYYY-MM-DD
    technician: str = ""
    storage_condition: str = "Unknown"  # "Room temp", "-20C", "-80C", etc.
    freeze_thaw_cycles: int = 0
    
    lysis: LysisMetadata = field(default_factory=LysisMetadata)
    protein_digestion: ProteinDigestionMetadata = field(default_factory=ProteinDigestionMetadata)
    alcohol_purification: AlcoholPurificationMetadata = field(default_factory=AlcoholPurificationMetadata)
    binding_chemistry: BindingChemistry = field(default_factory=BindingChemistry)
    elution: ElutionMetadata = field(default_factory=ElutionMetadata)
    qc: PostExtractionQC = field(default_factory=PostExtractionQC)


@dataclass
class LibraryPrepMetadata:
    """Library preparation metadata."""
    kit_name: str = ""
    kit_version: str = ""
    input_dna_amount_ng: Optional[float] = None
    fragmentation_method: str = ""  # "Sonication", "Enzymatic", "Thermal"
    fragmentation_duration_sec: Optional[float] = None
    pcr_polymerase: str = ""
    pcr_cycles: Optional[int] = None
    adapter_barcode_kit: str = ""
    cleanup_bead_ratio: str = ""  # e.g., "1:1", "1:0.8"
    library_normalization_method: str = ""  # "Equimolar", "Weighted", "None"
    target_fragment_size_bp: Optional[int] = None
    measured_fragment_size_bp: Optional[int] = None
    library_concentration_nm: Optional[float] = None
    qc_pass_fail: str = "Unknown"
    notes: str = ""


@dataclass
class SequencingMetadata:
    """Sequencing run metadata."""
    platform: str = ""  # "Illumina", "PacBio", "Oxford Nanopore", etc.
    instrument_model: str = ""
    flowcell_id: str = ""
    read_length_bp: Optional[int] = None
    read_type: str = "Paired-end"  # "Paired-end", "Single-end"
    lane: str = ""
    run_date: Optional[str] = None  # ISO format: YYYY-MM-DD
    sequencing_depth_target_million: Optional[float] = None
    demultiplexing_software: str = ""
    demultiplexing_version: str = ""
    basecalling_software: str = ""
    basecalling_version: str = ""
    operator: str = ""
    run_notes: str = ""


@dataclass
class BatchMetadata:
    """Complete batch metadata combining all wet-lab context."""
    batch_id: str = ""
    batch_name: str = ""
    batch_date: Optional[str] = None  # ISO format: YYYY-MM-DD
    
    # Metadata sections
    extraction: ExtractionMetadata = field(default_factory=ExtractionMetadata)
    library_prep: LibraryPrepMetadata = field(default_factory=LibraryPrepMetadata)
    sequencing: SequencingMetadata = field(default_factory=SequencingMetadata)
    
    # Per-sample overrides
    sample_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Operator notes
    operator_notes: str = ""
    
    # QC traceability
    qc_flags: List[str] = field(default_factory=list)  # ['contamination_detected', 'low_yield', etc.]
    qc_pass_overall: str = "Unknown"  # "Pass", "Fail", "Conditional"
    
    # Metadata tracking
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    

# ═══════════════════════════════════════════════════════════════════════════════
# QC VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════

class QCValidator:
    """Validates wet-lab QC metrics against expected ranges and flags anomalies."""
    
    # QC reference ranges (conservative, for research use)
    QC_REFERENCE = {
        "nanodrop_260_280_min": 1.7,
        "nanodrop_260_280_max": 2.0,
        "nanodrop_260_230_min": 2.0,
        "nanodrop_260_230_max": 2.5,
        "qubit_min_ng_ul": 10.0,
        "fragment_integrity_min_percent": 80.0,
        "pcr_cycles_max": 25,
        "pcr_cycles_ideal": 15,
    }
    
    @staticmethod
    def check_extraction_qc(qc: PostExtractionQC) -> List[str]:
        """Flag abnormal extraction QC values."""
        flags = []
        
        if qc.nanodrop_260_280 is not None:
            if qc.nanodrop_260_280 < QCValidator.QC_REFERENCE["nanodrop_260_280_min"]:
                flags.append("low_260_280_ratio_protein_contamination")
            elif qc.nanodrop_260_280 > QCValidator.QC_REFERENCE["nanodrop_260_280_max"]:
                flags.append("high_260_280_ratio_rna_contamination")
        
        if qc.nanodrop_260_230 is not None:
            if qc.nanodrop_260_230 < QCValidator.QC_REFERENCE["nanodrop_260_230_min"]:
                flags.append("low_260_230_ratio_salt_contamination")
        
        if qc.qubit_concentration_ng_ul is not None:
            if qc.qubit_concentration_ng_ul < QCValidator.QC_REFERENCE["qubit_min_ng_ul"]:
                flags.append("low_dna_yield")
        
        if qc.fragment_integrity_percent is not None:
            if qc.fragment_integrity_percent < QCValidator.QC_REFERENCE["fragment_integrity_min_percent"]:
                flags.append("degraded_dna_low_integrity")
        
        return flags
    
    @staticmethod
    def check_library_prep_qc(lib: LibraryPrepMetadata) -> List[str]:
        """Flag abnormal library prep parameters."""
        flags = []
        
        if lib.pcr_cycles is not None:
            if lib.pcr_cycles > QCValidator.QC_REFERENCE["pcr_cycles_max"]:
                flags.append("excess_pcr_cycles_amplification_bias")
        
        if lib.library_concentration_nm is not None and lib.library_concentration_nm < 2.0:
            flags.append("low_library_concentration")
        
        return flags
    
    @staticmethod
    def check_extraction_chemistry(ext: ExtractionMetadata) -> List[str]:
        """Flag missing or suspicious extraction parameters."""
        flags = []
        
        if not ext.extraction_kit:
            flags.append("extraction_kit_not_specified")
        
        if ext.freeze_thaw_cycles > 5:
            flags.append("excessive_freeze_thaw_dna_degradation")
        
        if ext.protein_digestion.concentration_ug_ml and ext.protein_digestion.concentration_ug_ml < 0.1:
            flags.append("low_proteinase_k_concentration")
        
        return flags


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CONSISTENCY VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class BatchConsistencyValidator:
    """Validates consistency across samples in a batch."""
    
    @staticmethod
    def validate_batch_uniformity(batch: BatchMetadata, sample_ids: List[str]) -> Dict[str, Any]:
        """
        Check if all samples in batch have consistent metadata.
        
        Returns:
            Dict with consistency status and flagged anomalies.
        """
        results = {
            "is_uniform": True,
            "flags": [],
            "anomaly_samples": {},
            "consistency_report": {}
        }
        
        # Check if critical fields are empty
        if not batch.extraction.extraction_kit:
            results["flags"].append("missing_extraction_kit")
        if not batch.library_prep.kit_name:
            results["flags"].append("missing_library_kit")
        if not batch.sequencing.platform:
            results["flags"].append("missing_sequencing_platform")
        
        # Check for per-sample overrides that differ significantly
        override_types = {}
        for sample_id, overrides in batch.sample_overrides.items():
            for key in overrides.keys():
                if key not in override_types:
                    override_types[key] = []
                override_types[key].append(sample_id)
        
        if len(override_types) > 0:
            results["is_uniform"] = False
            results["flags"].append(f"per_sample_overrides_detected:{len(override_types)}_fields")
            results["anomaly_samples"] = override_types
        
        return results
    
    @staticmethod
    def generate_consistency_heatmap_data(
        batch: BatchMetadata, 
        sample_ids: List[str]
    ) -> Dict[str, Any]:
        """Generate data suitable for heatmap visualization of batch uniformity."""
        consistency_scores = {}
        
        for sample_id in sample_ids:
            sample_overrides = batch.sample_overrides.get(sample_id, {})
            override_count = len(sample_overrides)
            # Consistency score: 100 if no overrides, decreases with overrides
            consistency_scores[sample_id] = max(0, 100 - (override_count * 10))
        
        return {
            "samples": sample_ids,
            "consistency_scores": consistency_scores,
            "uniformity_percent": (sum(consistency_scores.values()) / (100 * len(sample_ids)) * 100) if sample_ids else 0
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE PROVENANCE TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimelineEvent:
    """Single event in sample provenance timeline."""
    event_type: str  # "collected", "stored", "extracted", "qc", "library_prep", "sequenced", "vcf_generated", "analyzed"
    timestamp: str  # ISO format
    status: str = "Pass"  # "Pass", "Warning", "Fail"
    notes: str = ""


class SampleProvenanceTimeline:
    """Builds and manages sample provenance timeline."""
    
    @staticmethod
    def build_timeline(batch: BatchMetadata, sample_id: str) -> List[TimelineEvent]:
        """Build timeline for a single sample."""
        events = []
        
        # Collection (inferred from batch date)
        if batch.batch_date:
            events.append(TimelineEvent(
                event_type="collected",
                timestamp=batch.batch_date,
                status="Pass" if batch.extraction.sample_type != "Unknown" else "Warning"
            ))
        
        # Storage
        if batch.extraction.storage_condition:
            events.append(TimelineEvent(
                event_type="stored",
                timestamp=batch.created_timestamp,
                status="Pass" if batch.extraction.freeze_thaw_cycles <= 2 else "Warning",
                notes=f"Freeze-thaw cycles: {batch.extraction.freeze_thaw_cycles}"
            ))
        
        # Extraction
        if batch.extraction.extraction_date:
            qc_flags = QCValidator.check_extraction_qc(batch.extraction.qc)
            status = "Pass" if not qc_flags and batch.extraction.qc.qc_pass_fail == "Pass" else "Warning"
            events.append(TimelineEvent(
                event_type="extracted",
                timestamp=batch.extraction.extraction_date,
                status=status,
                notes=f"Kit: {batch.extraction.extraction_kit}"
            ))
        
        # QC measurement
        if batch.extraction.qc.nanodrop_260_280 or batch.extraction.qc.qubit_concentration_ng_ul:
            events.append(TimelineEvent(
                event_type="qc",
                timestamp=batch.extraction.extraction_date or batch.created_timestamp,
                status=batch.extraction.qc.qc_pass_fail,
                notes=f"260/280: {batch.extraction.qc.nanodrop_260_280}, Qubit: {batch.extraction.qc.qubit_concentration_ng_ul} ng/ul"
            ))
        
        # Library prep
        if batch.library_prep.kit_name:
            events.append(TimelineEvent(
                event_type="library_prep",
                timestamp=batch.created_timestamp,
                status=batch.library_prep.qc_pass_fail,
                notes=f"Kit: {batch.library_prep.kit_name}, PCR cycles: {batch.library_prep.pcr_cycles}"
            ))
        
        # Sequencing
        if batch.sequencing.run_date:
            events.append(TimelineEvent(
                event_type="sequenced",
                timestamp=batch.sequencing.run_date,
                status="Pass",
                notes=f"Platform: {batch.sequencing.platform}, Depth: {batch.sequencing.sequencing_depth_target_million}M"
            ))
        
        return events


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def batch_to_dict(batch: BatchMetadata) -> Dict[str, Any]:
    """Convert BatchMetadata to dictionary for serialization."""
    return asdict(batch)


def dict_to_batch(data: Dict[str, Any]) -> BatchMetadata:
    """Convert dictionary back to BatchMetadata (partial reconstruction)."""
    batch = BatchMetadata(
        batch_id=data.get("batch_id", ""),
        batch_name=data.get("batch_name", ""),
        batch_date=data.get("batch_date"),
        extraction=ExtractionMetadata(**data.get("extraction", {})) if data.get("extraction") else ExtractionMetadata(),
        library_prep=LibraryPrepMetadata(**data.get("library_prep", {})) if data.get("library_prep") else LibraryPrepMetadata(),
        sequencing=SequencingMetadata(**data.get("sequencing", {})) if data.get("sequencing") else SequencingMetadata(),
        sample_overrides=data.get("sample_overrides", {}),
        operator_notes=data.get("operator_notes", ""),
        qc_flags=data.get("qc_flags", []),
        qc_pass_overall=data.get("qc_pass_overall", "Unknown"),
    )
    return batch


def batch_to_json(batch: BatchMetadata) -> str:
    """Serialize BatchMetadata to JSON."""
    return json.dumps(batch_to_dict(batch), indent=2, default=str)


def batch_from_json(json_str: str) -> BatchMetadata:
    """Deserialize BatchMetadata from JSON."""
    data = json.loads(json_str)
    return dict_to_batch(data)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH DEFAULTS & TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_KIT_TEMPLATES = {
    "DNeasy Blood & Tissue Kit": ExtractionMetadata(
        extraction_kit="DNeasy Blood & Tissue Kit",
        lysis=LysisMetadata(buffer_type="Tris-HCl", incubation_temperature_c=56),
        binding_chemistry=BindingChemistry(method="Silica column", column_type="DNeasy spin column")
    ),
    "CTAB Extraction": ExtractionMetadata(
        extraction_kit="CTAB (Cetyltrimethylammonium Bromide)",
        lysis=LysisMetadata(buffer_type="CTAB buffer", incubation_temperature_c=65),
        binding_chemistry=BindingChemistry(method="Precipitation", custom_notes="Chloroform-isoamyl alcohol extraction")
    ),
    "Phenol-Chloroform": ExtractionMetadata(
        extraction_kit="Phenol-Chloroform Extraction",
        lysis=LysisMetadata(buffer_type="TE buffer + SDS"),
        binding_chemistry=BindingChemistry(method="Precipitation", custom_notes="Classic phenol-chloroform method")
    ),
}

LIBRARY_KIT_TEMPLATES = {
    "Illumina TruSeq DNA": LibraryPrepMetadata(
        kit_name="Illumina TruSeq DNA Library Prep Kit",
        fragmentation_method="Sonication",
        pcr_cycles=15,
        target_fragment_size_bp=450
    ),
    "NEBNext Ultra II": LibraryPrepMetadata(
        kit_name="NEBNext Ultra II DNA Library Prep Kit",
        fragmentation_method="Enzymatic",
        pcr_cycles=12,
        target_fragment_size_bp=550
    ),
    "KAPA Hyper": LibraryPrepMetadata(
        kit_name="KAPA Hyper Prep Kit",
        fragmentation_method="Sonication",
        pcr_cycles=13,
        target_fragment_size_bp=500
    ),
}

SEQUENCING_PLATFORM_TEMPLATES = {
    "Illumina NovaSeq 6000": SequencingMetadata(
        platform="Illumina",
        instrument_model="NovaSeq 6000",
        read_type="Paired-end",
        basecalling_software="RTA3",
        demultiplexing_software="bcl2fastq2"
    ),
    "Illumina NextSeq 550": SequencingMetadata(
        platform="Illumina",
        instrument_model="NextSeq 550",
        read_type="Paired-end",
        basecalling_software="RTA2",
        demultiplexing_software="bcl2fastq2"
    ),
    "PacBio Sequel II": SequencingMetadata(
        platform="PacBio",
        instrument_model="Sequel II",
        read_type="Single-end",
        basecalling_software="SMRT Link",
    ),
}


def create_batch_from_template(
    batch_id: str,
    extraction_template: Optional[str] = None,
    library_template: Optional[str] = None,
    sequencing_template: Optional[str] = None
) -> BatchMetadata:
    """Create a BatchMetadata from predefined templates."""
    batch = BatchMetadata(batch_id=batch_id)
    
    if extraction_template and extraction_template in EXTRACTION_KIT_TEMPLATES:
        batch.extraction = EXTRACTION_KIT_TEMPLATES[extraction_template]
    
    if library_template and library_template in LIBRARY_KIT_TEMPLATES:
        batch.library_prep = LIBRARY_KIT_TEMPLATES[library_template]
    
    if sequencing_template and sequencing_template in SEQUENCING_PLATFORM_TEMPLATES:
        batch.sequencing = SEQUENCING_PLATFORM_TEMPLATES[sequencing_template]
    
    return batch
