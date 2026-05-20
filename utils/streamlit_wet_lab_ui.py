"""
Streamlit UI Components for Wet-Lab Batch Context

Provides reusable Streamlit components for capturing, displaying, and interpreting
wet-lab metadata in the variant analysis dashboard.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from utils.wet_lab_provenance import (
    BatchMetadata, ExtractionMetadata, LibraryPrepMetadata, SequencingMetadata,
    PostExtractionQC, ProteinDigestionMetadata, LysisMetadata, ElutionMetadata,
    BindingChemistry, AlcoholPurificationMetadata,
    QCValidator, BatchConsistencyValidator, SampleProvenanceTimeline,
    EXTRACTION_KIT_TEMPLATES, LIBRARY_KIT_TEMPLATES, SEQUENCING_PLATFORM_TEMPLATES,
    create_batch_from_template
)
from utils.batch_inference import (
    DownstreamInferenceEngine, RiskSeverity, VariantConfidenceAssessor
)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH INITIALIZATION & MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_batch_session_state() -> None:
    """Initialize session state for batch metadata capture."""
    if "batch_metadata" not in st.session_state:
        st.session_state.batch_metadata = BatchMetadata()
    
    if "batch_mode_enabled" not in st.session_state:
        st.session_state.batch_mode_enabled = False
    
    if "batch_sample_ids" not in st.session_state:
        st.session_state.batch_sample_ids = []
    
    if "batch_risk_flags" not in st.session_state:
        st.session_state.batch_risk_flags = []


def get_batch_from_session() -> BatchMetadata:
    """Retrieve batch metadata from session state."""
    if "batch_metadata" not in st.session_state:
        initialize_batch_session_state()
    return st.session_state.batch_metadata


def save_batch_to_session(batch: BatchMetadata) -> None:
    """Save batch metadata to session state."""
    st.session_state.batch_metadata = batch


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH OVERVIEW PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def render_batch_overview(batch: Optional[BatchMetadata] = None) -> None:
    """Render batch overview summary card."""
    if batch is None:
        batch = get_batch_from_session()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Batch ID",
            batch.batch_id if batch.batch_id else "Not set",
            help="Unique identifier for this batch"
        )
    
    with col2:
        batch_date = batch.batch_date if batch.batch_date else "Unknown"
        st.metric("Batch Date", batch_date)
    
    with col3:
        sample_count = len(batch.sample_overrides)
        st.metric("Samples", sample_count, help="Number of samples in this batch")
    
    with col4:
        risk_level = "🟢 Low Risk" if batch.qc_pass_overall == "Pass" else \
                     "🟡 Moderate Risk" if batch.qc_pass_overall == "Unknown" else \
                     "🔴 High Risk"
        st.metric("QC Status", risk_level)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION METADATA CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

def render_extraction_input(batch: Optional[BatchMetadata] = None) -> BatchMetadata:
    """Render DNA extraction metadata input form."""
    if batch is None:
        batch = get_batch_from_session()
    
    st.markdown("### 🧪 DNA Extraction Metadata")
    st.caption("Capture upstream laboratory chemistry that affects variant quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Template selector
        kit_template = st.selectbox(
            "Extraction Kit Template",
            ["None"] + list(EXTRACTION_KIT_TEMPLATES.keys()),
            help="Select a predefined template to auto-fill parameters"
        )
        if kit_template != "None":
            batch.extraction = EXTRACTION_KIT_TEMPLATES[kit_template]
            st.success(f"Loaded template: {kit_template}")
        
        batch.extraction.extraction_kit = st.text_input(
            "Extraction Kit",
            value=batch.extraction.extraction_kit,
            placeholder="e.g., DNeasy Blood & Tissue Kit",
            help="Name of the extraction kit used"
        )
        
        batch.extraction.sample_type = st.selectbox(
            "Sample Type",
            ["Unknown", "Whole blood", "Tissue", "Cell line", "Plasma", "Saliva", "Buccal"],
            index=0 if batch.extraction.sample_type == "Unknown" else 
                   ["Unknown", "Whole blood", "Tissue", "Cell line", "Plasma", "Saliva", "Buccal"].index(batch.extraction.sample_type)
        )
        
        batch.extraction.extraction_date = st.date_input(
            "Extraction Date",
            value=datetime.now() if not batch.extraction.extraction_date else datetime.fromisoformat(batch.extraction.extraction_date)
        ).isoformat() if st.session_state.get("_extract_date_set", False) or batch.extraction.extraction_date else None
        
        # Trigger to set extraction date
        if st.checkbox("Set extraction date", value=bool(batch.extraction.extraction_date)):
            st.session_state["_extract_date_set"] = True
            batch.extraction.extraction_date = st.date_input(
                "Extraction Date",
                value=datetime.now() if not batch.extraction.extraction_date else datetime.fromisoformat(batch.extraction.extraction_date)
            ).isoformat()
    
    with col2:
        batch.extraction.technician = st.text_input(
            "Technician",
            value=batch.extraction.technician,
            placeholder="Operator name"
        )
        
        batch.extraction.storage_condition = st.selectbox(
            "Storage Condition",
            ["Unknown", "Room temperature", "-20°C", "-80°C", "Liquid nitrogen"],
            index=0 if batch.extraction.storage_condition == "Unknown" else
                   ["Unknown", "Room temperature", "-20°C", "-80°C", "Liquid nitrogen"].index(batch.extraction.storage_condition)
        )
        
        batch.extraction.freeze_thaw_cycles = st.number_input(
            "Freeze-Thaw Cycles",
            min_value=0, max_value=10, value=batch.extraction.freeze_thaw_cycles,
            help="Number of freeze-thaw cycles (≤2 recommended)"
        )
    
    # Lysis parameters
    st.markdown("#### Lysis Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch.extraction.lysis.buffer_type = st.text_input(
            "Lysis Buffer Type",
            value=batch.extraction.lysis.buffer_type,
            placeholder="e.g., Tris-HCl"
        )
    with col2:
        batch.extraction.lysis.incubation_temperature_c = st.number_input(
            "Temperature (°C)",
            value=batch.extraction.lysis.incubation_temperature_c or 56,
            min_value=0, max_value=100
        )
    with col3:
        batch.extraction.lysis.incubation_duration_min = st.number_input(
            "Duration (min)",
            value=batch.extraction.lysis.incubation_duration_min or 10,
            min_value=0
        )
    
    # Protein digestion
    st.markdown("#### Protein Digestion")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        batch.extraction.protein_digestion.concentration_ug_ml = st.number_input(
            "Proteinase K (µg/mL)",
            value=batch.extraction.protein_digestion.concentration_ug_ml or 0.5,
            min_value=0.0, step=0.1
        )
    with col2:
        batch.extraction.protein_digestion.incubation_temperature_c = st.number_input(
            "Temperature (°C)",
            value=batch.extraction.protein_digestion.incubation_temperature_c or 56,
            min_value=0, max_value=100
        )
    with col3:
        batch.extraction.protein_digestion.incubation_duration_min = st.number_input(
            "Duration (min)",
            value=batch.extraction.protein_digestion.incubation_duration_min or 30,
            min_value=0
        )
    with col4:
        batch.extraction.protein_digestion.enzyme_lot_number = st.text_input(
            "Enzyme Lot",
            value=batch.extraction.protein_digestion.enzyme_lot_number,
            placeholder="e.g., P8107S"
        )
    
    # Alcohol purification
    st.markdown("#### Alcohol Purification")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch.extraction.alcohol_purification.alcohol_type = st.selectbox(
            "Alcohol Type",
            ["Ethanol", "Isopropanol"],
            index=0 if batch.extraction.alcohol_purification.alcohol_type == "Ethanol" else 1
        )
    with col2:
        batch.extraction.alcohol_purification.concentration_percent = st.number_input(
            "Concentration (%)",
            value=batch.extraction.alcohol_purification.concentration_percent or 100.0,
            min_value=0.0, max_value=100.0
        )
    with col3:
        batch.extraction.alcohol_purification.wash_count = st.number_input(
            "Wash Count",
            value=batch.extraction.alcohol_purification.wash_count,
            min_value=1, max_value=10
        )
    
    # Binding chemistry
    st.markdown("#### Binding Chemistry")
    binding_method = st.selectbox(
        "Binding Method",
        ["Silica column", "Magnetic bead", "Other"],
        index=0 if batch.extraction.binding_chemistry.method == "Silica column" else
               1 if batch.extraction.binding_chemistry.method == "Magnetic bead" else 2
    )
    batch.extraction.binding_chemistry.method = binding_method
    
    if binding_method == "Silica column":
        batch.extraction.binding_chemistry.column_type = st.text_input(
            "Column Type",
            value=batch.extraction.binding_chemistry.column_type,
            placeholder="e.g., DNeasy spin column"
        )
    elif binding_method == "Magnetic bead":
        batch.extraction.binding_chemistry.magnetic_bead_type = st.text_input(
            "Magnetic Bead Type",
            value=batch.extraction.binding_chemistry.magnetic_bead_type,
            placeholder="e.g., Dynabeads M-280"
        )
    else:
        batch.extraction.binding_chemistry.custom_notes = st.text_area(
            "Method Notes",
            value=batch.extraction.binding_chemistry.custom_notes,
            height=100
        )
    
    # Elution
    st.markdown("#### Elution")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch.extraction.elution.buffer_type = st.text_input(
            "Elution Buffer",
            value=batch.extraction.elution.buffer_type,
            placeholder="e.g., TE buffer"
        )
    with col2:
        batch.extraction.elution.buffer_volume_ul = st.number_input(
            "Volume (µL)",
            value=batch.extraction.elution.buffer_volume_ul or 100,
            min_value=0.0
        )
    with col3:
        batch.extraction.elution.elution_repetitions = st.number_input(
            "Repetitions",
            value=batch.extraction.elution.elution_repetitions,
            min_value=1, max_value=5
        )
    
    # Post-extraction QC
    st.markdown("#### Post-Extraction QC")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        batch.extraction.qc.nanodrop_260_280 = st.number_input(
            "260/280 Ratio",
            value=batch.extraction.qc.nanodrop_260_280 or 0.0,
            min_value=0.0, step=0.1,
            help="Expect 1.7–2.0; <1.7 indicates protein contamination"
        )
    with col2:
        batch.extraction.qc.nanodrop_260_230 = st.number_input(
            "260/230 Ratio",
            value=batch.extraction.qc.nanodrop_260_230 or 0.0,
            min_value=0.0, step=0.1,
            help="Expect >2.0; <2.0 indicates salt contamination"
        )
    with col3:
        batch.extraction.qc.qubit_concentration_ng_ul = st.number_input(
            "Qubit (ng/µL)",
            value=batch.extraction.qc.qubit_concentration_ng_ul or 0.0,
            min_value=0.0,
            help="DNA concentration; expect ≥10 ng/µL"
        )
    with col4:
        batch.extraction.qc.fragment_integrity_percent = st.number_input(
            "Fragment Integrity (%)",
            value=batch.extraction.qc.fragment_integrity_percent or 0.0,
            min_value=0.0, max_value=100.0,
            help="Expect ≥80%; <80% indicates degradation"
        )
    
    batch.extraction.qc.qc_pass_fail = st.selectbox(
        "QC Pass/Fail",
        ["Unknown", "Pass", "Fail", "Warning"],
        index=0 if batch.extraction.qc.qc_pass_fail == "Unknown" else
               ["Unknown", "Pass", "Fail", "Warning"].index(batch.extraction.qc.qc_pass_fail)
    )
    
    # Flag abnormal values
    if batch.extraction.qc.nanodrop_260_280 or batch.extraction.qc.qubit_concentration_ng_ul:
        qc_flags = QCValidator.check_extraction_qc(batch.extraction.qc)
        if qc_flags:
            st.warning(f"⚠️ QC Flags: {', '.join(qc_flags)}")
    
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARY PREP METADATA CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

def render_library_prep_input(batch: Optional[BatchMetadata] = None) -> BatchMetadata:
    """Render library preparation metadata input form."""
    if batch is None:
        batch = get_batch_from_session()
    
    st.markdown("### 📚 Library Preparation Metadata")
    st.caption("Library kit, fragmentation, and PCR parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Template selector
        lib_template = st.selectbox(
            "Library Kit Template",
            ["None"] + list(LIBRARY_KIT_TEMPLATES.keys()),
            help="Select a predefined template to auto-fill parameters",
            key="lib_template_selector"
        )
        if lib_template != "None":
            batch.library_prep = LIBRARY_KIT_TEMPLATES[lib_template]
            st.success(f"Loaded template: {lib_template}")
        
        batch.library_prep.kit_name = st.text_input(
            "Library Kit Name",
            value=batch.library_prep.kit_name,
            placeholder="e.g., Illumina TruSeq DNA"
        )
        
        batch.library_prep.kit_version = st.text_input(
            "Kit Version",
            value=batch.library_prep.kit_version,
            placeholder="e.g., Standard"
        )
        
        batch.library_prep.input_dna_amount_ng = st.number_input(
            "Input DNA Amount (ng)",
            value=batch.library_prep.input_dna_amount_ng or 0.0,
            min_value=0.0
        )
    
    with col2:
        batch.library_prep.fragmentation_method = st.selectbox(
            "Fragmentation Method",
            ["Sonication", "Enzymatic", "Thermal", "None"],
            index=0 if batch.library_prep.fragmentation_method in ["Sonication", "Enzymatic", "Thermal", "None"] else 0
        )
        
        if batch.library_prep.fragmentation_method != "None":
            batch.library_prep.fragmentation_duration_sec = st.number_input(
                "Fragmentation Duration (sec)",
                value=batch.library_prep.fragmentation_duration_sec or 0.0,
                min_value=0.0
            )
        
        batch.library_prep.target_fragment_size_bp = st.number_input(
            "Target Fragment Size (bp)",
            value=batch.library_prep.target_fragment_size_bp or 500,
            min_value=0
        )
        
        batch.library_prep.measured_fragment_size_bp = st.number_input(
            "Measured Fragment Size (bp)",
            value=batch.library_prep.measured_fragment_size_bp or 0,
            min_value=0
        )
    
    # PCR parameters
    st.markdown("#### PCR Amplification")
    col1, col2, col3 = st.columns(3)
    with col1:
        batch.library_prep.pcr_polymerase = st.text_input(
            "PCR Polymerase",
            value=batch.library_prep.pcr_polymerase,
            placeholder="e.g., Q5 High-Fidelity"
        )
    with col2:
        batch.library_prep.pcr_cycles = st.number_input(
            "PCR Cycles",
            value=batch.library_prep.pcr_cycles or 15,
            min_value=1, max_value=35,
            help="Typical: 12–18; >25 risks amplification bias"
        )
    with col3:
        batch.library_prep.adapter_barcode_kit = st.text_input(
            "Adapter/Barcode Kit",
            value=batch.library_prep.adapter_barcode_kit,
            placeholder="e.g., NEXTflex"
        )
    
    # Quality
    st.markdown("#### Library QC")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        batch.library_prep.library_concentration_nm = st.number_input(
            "Library Concentration (nM)",
            value=batch.library_prep.library_concentration_nm or 0.0,
            min_value=0.0,
            help="Expect ≥2 nM"
        )
    with col2:
        batch.library_prep.cleanup_bead_ratio = st.text_input(
            "Cleanup Bead Ratio",
            value=batch.library_prep.cleanup_bead_ratio,
            placeholder="e.g., 1:1"
        )
    with col3:
        batch.library_prep.library_normalization_method = st.selectbox(
            "Normalization",
            ["None", "Equimolar", "Weighted", "Other"],
            index=0 if batch.library_prep.library_normalization_method == "None" else 1
        )
    with col4:
        batch.library_prep.qc_pass_fail = st.selectbox(
            "QC Pass/Fail",
            ["Unknown", "Pass", "Fail", "Warning"],
            index=0 if batch.library_prep.qc_pass_fail == "Unknown" else
                   ["Unknown", "Pass", "Fail", "Warning"].index(batch.library_prep.qc_pass_fail),
            key="lib_qc_pass_fail"
        )
    
    # Flag abnormal values
    if batch.library_prep.pcr_cycles or batch.library_prep.library_concentration_nm:
        lib_flags = QCValidator.check_library_prep_qc(batch.library_prep)
        if lib_flags:
            st.warning(f"⚠️ Library Flags: {', '.join(lib_flags)}")
    
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCING METADATA CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

def render_sequencing_input(batch: Optional[BatchMetadata] = None) -> BatchMetadata:
    """Render sequencing metadata input form."""
    if batch is None:
        batch = get_batch_from_session()
    
    st.markdown("### 🧬 Sequencing Metadata")
    st.caption("Sequencing run parameters and instrument details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Template selector
        seq_template = st.selectbox(
            "Sequencing Platform Template",
            ["None"] + list(SEQUENCING_PLATFORM_TEMPLATES.keys()),
            help="Select a predefined template to auto-fill parameters",
            key="seq_template_selector"
        )
        if seq_template != "None":
            batch.sequencing = SEQUENCING_PLATFORM_TEMPLATES[seq_template]
            st.success(f"Loaded template: {seq_template}")
        
        batch.sequencing.platform = st.selectbox(
            "Platform",
            ["Illumina", "PacBio", "Oxford Nanopore", "Ion Torrent", "Other"],
            index=0 if batch.sequencing.platform == "Illumina" else 1 if batch.sequencing.platform == "PacBio" else 2,
            key="seq_platform"
        )
        
        batch.sequencing.instrument_model = st.text_input(
            "Instrument Model",
            value=batch.sequencing.instrument_model,
            placeholder="e.g., NovaSeq 6000"
        )
        
        batch.sequencing.flowcell_id = st.text_input(
            "Flowcell ID",
            value=batch.sequencing.flowcell_id,
            placeholder="e.g., H00WJG45V01"
        )
    
    with col2:
        batch.sequencing.read_length_bp = st.number_input(
            "Read Length (bp)",
            value=batch.sequencing.read_length_bp or 150,
            min_value=0
        )
        
        batch.sequencing.read_type = st.selectbox(
            "Read Type",
            ["Paired-end", "Single-end"],
            index=0 if batch.sequencing.read_type == "Paired-end" else 1
        )
        
        batch.sequencing.lane = st.text_input(
            "Lane",
            value=batch.sequencing.lane,
            placeholder="e.g., 1"
        )
        
        batch.sequencing.run_date = st.date_input(
            "Run Date",
            value=datetime.now() if not batch.sequencing.run_date else datetime.fromisoformat(batch.sequencing.run_date)
        ).isoformat() if st.session_state.get("_seq_date_set", False) or batch.sequencing.run_date else None
        
        if st.checkbox("Set sequencing run date", value=bool(batch.sequencing.run_date)):
            st.session_state["_seq_date_set"] = True
            batch.sequencing.run_date = st.date_input(
                "Run Date",
                value=datetime.now() if not batch.sequencing.run_date else datetime.fromisoformat(batch.sequencing.run_date)
            ).isoformat()
    
    # Sequencing depth
    st.markdown("#### Sequencing Depth")
    col1, col2 = st.columns(2)
    with col1:
        batch.sequencing.sequencing_depth_target_million = st.number_input(
            "Target Depth (Million reads)",
            value=batch.sequencing.sequencing_depth_target_million or 30.0,
            min_value=0.0,
            help="WGS: ≥30M; WES: ≥100M; panels: ≥1000x"
        )
    with col2:
        st.info("Actual depth will be measured from BAM file post-alignment")
    
    # Software versions
    st.markdown("#### Software Versions")
    col1, col2 = st.columns(2)
    with col1:
        batch.sequencing.basecalling_software = st.text_input(
            "Basecalling Software",
            value=batch.sequencing.basecalling_software,
            placeholder="e.g., RTA3"
        )
        batch.sequencing.basecalling_version = st.text_input(
            "Basecalling Version",
            value=batch.sequencing.basecalling_version,
            placeholder="e.g., 3.4.4"
        )
    with col2:
        batch.sequencing.demultiplexing_software = st.text_input(
            "Demultiplexing Software",
            value=batch.sequencing.demultiplexing_software,
            placeholder="e.g., bcl2fastq2"
        )
        batch.sequencing.demultiplexing_version = st.text_input(
            "Demultiplexing Version",
            value=batch.sequencing.demultiplexing_version,
            placeholder="e.g., 2.20"
        )
    
    # Operator
    st.markdown("#### Run Operator")
    batch.sequencing.operator = st.text_input(
        "Operator Name",
        value=batch.sequencing.operator,
        placeholder="Name of sequencing operator"
    )
    
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# RISK FLAGS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def render_wet_lab_risk_flags(batch: Optional[BatchMetadata] = None) -> None:
    """Render wet-lab risk flags derived from metadata."""
    if batch is None:
        batch = get_batch_from_session()
    
    st.markdown("### 🚨 Wet-Lab Risk Flags")
    st.caption("Downstream impact predictions based on extraction & library metadata")
    
    flags = DownstreamInferenceEngine.infer_risks(batch)
    batch_modifier = DownstreamInferenceEngine.compute_batch_confidence_modifier(batch)
    
    if not flags:
        st.success("✅ No significant risk flags detected.")
    else:
        # Summary
        critical_count = sum(1 for f in flags if f.severity == RiskSeverity.CRITICAL)
        warning_count = sum(1 for f in flags if f.severity == RiskSeverity.WARNING)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Critical", critical_count)
        col2.metric("⚠️ Warning", warning_count)
        col3.metric("📊 Batch Confidence Modifier", f"{batch_modifier:.2%}")
        
        st.divider()
        
        # Display flags
        for flag in flags:
            with st.expander(flag.title, expanded=flag.severity == RiskSeverity.CRITICAL):
                st.markdown(DownstreamInferenceEngine.flag_to_markdown(flag))
    
    # Store in session
    st.session_state.batch_risk_flags = flags


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CONSISTENCY PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def render_batch_consistency(batch: Optional[BatchMetadata] = None, sample_ids: Optional[List[str]] = None) -> None:
    """Render batch consistency validation panel."""
    if batch is None:
        batch = get_batch_from_session()
    if sample_ids is None:
        sample_ids = st.session_state.get("batch_sample_ids", [])
    
    st.markdown("### 📊 Batch Consistency Validation")
    st.caption("Check for suspicious variations across samples")
    
    validation_result = BatchConsistencyValidator.validate_batch_uniformity(batch, sample_ids)
    uniformity_data = BatchConsistencyValidator.generate_consistency_heatmap_data(batch, sample_ids)
    
    if validation_result["is_uniform"]:
        st.success("✅ Batch is uniform across all samples")
    else:
        st.warning(f"⚠️ {len(validation_result['flags'])} consistency flags detected")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Uniformity Score", f"{uniformity_data['uniformity_percent']:.1f}%")
    with col2:
        st.metric("Per-Sample Overrides", len(batch.sample_overrides))
    
    if validation_result["flags"]:
        st.subheader("Flags")
        for flag in validation_result["flags"]:
            st.markdown(f"- {flag}")
    
    if uniformity_data["consistency_scores"]:
        # Create a simple consistency table
        df_consistency = pd.DataFrame({
            "Sample": uniformity_data["samples"],
            "Consistency Score": [uniformity_data["consistency_scores"].get(s, 100) for s in uniformity_data["samples"]]
        })
        st.dataframe(df_consistency, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE PROVENANCE TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

def render_sample_timeline(batch: Optional[BatchMetadata] = None, sample_id: Optional[str] = None) -> None:
    """Render sample provenance timeline visualization."""
    if batch is None:
        batch = get_batch_from_session()
    if sample_id is None:
        sample_id = "Sample 1"
    
    st.markdown(f"### ⏱️ Sample Provenance Timeline: {sample_id}")
    
    timeline_events = SampleProvenanceTimeline.build_timeline(batch, sample_id)
    
    if not timeline_events:
        st.info("No timeline events available for this sample.")
        return
    
    # Render as a vertical timeline
    for i, event in enumerate(timeline_events):
        col1, col2, col3 = st.columns([1, 2, 8])
        
        status_emoji = "✅" if event.status == "Pass" else "⚠️" if event.status == "Warning" else "❌"
        
        with col1:
            st.markdown(f"**{i+1}**")
        with col2:
            st.markdown(f"{status_emoji} {event.event_type.replace('_', ' ').title()}")
        with col3:
            st.markdown(f"**{event.timestamp}** — {event.notes}")
    
    return timeline_events


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT CONFIDENCE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def render_variant_wet_lab_context(
    variant_quality: float,
    variant_depth: float,
    batch: Optional[BatchMetadata] = None
) -> None:
    """Render wet-lab context panel for a single variant."""
    if batch is None:
        batch = get_batch_from_session()
    
    flags = DownstreamInferenceEngine.infer_risks(batch)
    batch_modifier = DownstreamInferenceEngine.compute_batch_confidence_modifier(batch)
    
    assessment = VariantConfidenceAssessor.assess_variant_confidence(
        variant_quality, variant_depth, batch_modifier, flags
    )
    
    st.markdown("#### Wet-Lab Context")
    
    # Color mapping
    color_map = {
        "high": "🟢",
        "moderate": "🟡",
        "low": "🔴",
        "critical": "🔴"
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Base Confidence", f"{assessment['base_confidence']:.1f}%")
    with col2:
        st.metric("Batch Modifier", f"{assessment['batch_modifier']:.2%}")
    with col3:
        emoji = color_map.get(assessment['color_class'], "•")
        st.markdown(f"#### {emoji} {assessment['interpretation_level']}")
    
    st.metric("Adjusted Confidence", f"{assessment['adjusted_confidence']:.1f}%")
    
    if assessment['notes']:
        st.info("\n".join(assessment['notes']))
    
    if flags:
        st.markdown("**Applicable Risk Flags:**")
        for flag in flags[:3]:  # Show top 3
            st.markdown(f"- {flag.title}")
        if len(flags) > 3:
            st.markdown(f"- *...and {len(flags) - 3} more*")


def render_collapsible_batch_context() -> None:
    """Render the main collapsible batch context section."""
    st.markdown("---")
    
    with st.expander("🔬 Wet-Lab Batch Context", expanded=True):
        initialize_batch_session_state()
        batch = get_batch_from_session()
        
        # Batch ID and basic info
        col1, col2 = st.columns(2)
        with col1:
            batch.batch_id = st.text_input(
                "Batch ID",
                value=batch.batch_id,
                placeholder="e.g., BATCH-2026-05-12-001",
                help="Unique identifier for this batch"
            )
        with col2:
            if st.checkbox("Enable Full Pipeline Mode"):
                st.session_state.batch_mode_enabled = True
                st.success("✅ Full pipeline mode enabled")
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Extraction",
            "Library Prep",
            "Sequencing",
            "Risk Assessment",
            "Batch QC"
        ])
        
        with tab1:
            batch = render_extraction_input(batch)
        
        with tab2:
            batch = render_library_prep_input(batch)
        
        with tab3:
            batch = render_sequencing_input(batch)
        
        with tab4:
            render_wet_lab_risk_flags(batch)
        
        with tab5:
            batch.qc_pass_overall = st.selectbox(
                "Overall Batch QC Status",
                ["Unknown", "Pass", "Fail", "Conditional"],
                index=0 if batch.qc_pass_overall == "Unknown" else
                       ["Unknown", "Pass", "Fail", "Conditional"].index(batch.qc_pass_overall)
            )
            batch.operator_notes = st.text_area(
                "Operator Notes",
                value=batch.operator_notes,
                height=150,
                placeholder="Document any anomalies, special handling, or quality concerns..."
            )
            render_batch_consistency(batch)
        
        # Save batch
        save_batch_to_session(batch)
        
        # Summary card
        st.divider()
        render_batch_overview(batch)
