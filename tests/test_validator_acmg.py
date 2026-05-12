"""Tests for utils/validator.py and utils/acmg.py."""
import io
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.validator import validate_vcf
from utils.acmg import classify_variant, classify_dataframe, _classify, classify_pm2_african, classify_ba1_african


# ── validator tests ───────────────────────────────────────────────────────────

MINIMAL_VCF = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
chr1\t12345\t.\tA\tT\t50\tPASS\tDP=30
"""

def _upload(content: bytes, size: int = None):
    buf = io.BytesIO(content)
    buf.name = "test.vcf"
    buf.size = size if size is not None else len(content)
    return buf


def test_valid_vcf_passes():
    ok, err = validate_vcf(_upload(MINIMAL_VCF))
    assert ok is True
    assert err == ""


def test_empty_file_fails():
    ok, err = validate_vcf(_upload(b""))
    assert ok is False
    assert "empty" in err.lower()


def test_missing_fileformat_fails():
    bad = b"##INFO=<ID=DP,Number=1>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t1\t.\tA\tT\t50\tPASS\t.\n"
    ok, err = validate_vcf(_upload(bad))
    assert ok is False
    assert "fileformat" in err.lower()


def test_missing_chrom_header_fails():
    bad = b"##fileformat=VCFv4.2\nchr1\t1\t.\tA\tT\t50\tPASS\t.\n"
    ok, err = validate_vcf(_upload(bad))
    assert ok is False
    assert "#CHROM" in err or "chrom" in err.lower()


def test_not_vcf_at_all_fails():
    ok, err = validate_vcf(_upload(b"This is a plain text file, not a VCF.\n"))
    assert ok is False


def test_file_too_large_fails():
    ok, err = validate_vcf(_upload(MINIMAL_VCF, size=600 * 1024 * 1024))
    assert ok is False
    assert "MB" in err


# ── ACMG tests ────────────────────────────────────────────────────────────────

def test_pvs1_lof_classified_likely_pathogenic():
    row = pd.Series({
        "variant_type": "SNP",
        "ref": "G", "alt": "A",
        "annotation": "stop_gained",
        "gnomad_af": None,
        "ClinVar Significance": "",
        "info_raw": "",
    })
    result = classify_variant(row)
    assert result["acmg_class"] in ("Likely Pathogenic", "Pathogenic")
    assert "PVS1" in result["acmg_path_evidence"]


def test_ba1_common_allele_benign():
    row = pd.Series({
        "variant_type": "SNP",
        "ref": "A", "alt": "G",
        "annotation": "missense_variant",
        "gnomad_af": 0.12,
        "ClinVar Significance": "",
        "info_raw": "",
    })
    result = classify_variant(row)
    assert result["acmg_class"] == "Benign"
    assert "BA1" in result["acmg_benign_evidence"]


def test_pm2_ultra_rare_vus():
    row = pd.Series({
        "variant_type": "SNP",
        "ref": "C", "alt": "T",
        "annotation": "missense_variant",
        "gnomad_af": 0.00005,
        "ClinVar Significance": "",
        "info_raw": "",
    })
    result = classify_variant(row)
    assert "PM2" in result["acmg_path_evidence"]


def test_clinvar_pathogenic_ps1():
    row = pd.Series({
        "variant_type": "SNP",
        "ref": "G", "alt": "A",
        "annotation": "missense_variant",
        "gnomad_af": None,
        "ClinVar Significance": "Pathogenic",
        "info_raw": "",
    })
    result = classify_variant(row)
    assert "PS1" in result["acmg_path_evidence"]


def test_classify_dataframe_adds_columns():
    df = pd.DataFrame([{
        "chrom": "chr1", "position": 100, "ref": "A", "alt": "T",
        "variant_type": "SNP", "annotation": "missense_variant",
        "gnomad_af": None, "ClinVar Significance": "", "info_raw": "",
    }])
    result = classify_dataframe(df)
    assert "acmg_class" in result.columns
    assert "acmg_path_evidence" in result.columns
    assert "acmg_benign_evidence" in result.columns


def test_classify_empty_dataframe():
    result = classify_dataframe(pd.DataFrame())
    assert result.empty


def test_inframe_indel_pm4():
    row = pd.Series({
        "variant_type": "INDEL",
        "ref": "AAAGGG", "alt": "AAA",
        "annotation": "inframe_deletion",
        "gnomad_af": None,
        "ClinVar Significance": "",
        "info_raw": "",
    })
    result = classify_variant(row)
    assert "PM4" in result["acmg_path_evidence"]


# ── African contextualisation tests ───────────────────────────────────────

def test_pm2_african_context_mismatch():
    """Test PM2 African contextualisation when African AF >> overall AF."""
    result = classify_pm2_african(overall_gnomad_af=0.0001, african_gnomad_af=0.02)
    assert result["standard_call"] == "PM2_Supporting"
    assert result["african_call"] == "Not_PM2"
    assert result["flag"] == "AFRICAN_CONTEXT_MISMATCH"
    assert "⚠️" in result["recommendation"]


def test_pm2_african_no_mismatch():
    """Test PM2 when African AF is proportional to overall AF."""
    result = classify_pm2_african(overall_gnomad_af=0.0001, african_gnomad_af=0.00008)
    assert result["standard_call"] == "PM2_Supporting"
    assert result["african_call"] == "PM2_Supporting"
    assert result["flag"] is None


def test_ba1_african_context_mismatch():
    """Test BA1 African contextualisation when African AF << overall AF."""
    result = classify_ba1_african(overall_gnomad_af=0.08, african_gnomad_af=0.001)
    assert result["standard_call"] == "BA1_Strong"
    assert result["african_call"] == "Not_BA1"
    assert result["flag"] == "AFRICAN_CONTEXT_MISMATCH"
    assert "⚠️" in result["recommendation"]


def test_ba1_african_no_mismatch():
    """Test BA1 when African AF is proportional to overall AF."""
    result = classify_ba1_african(overall_gnomad_af=0.08, african_gnomad_af=0.06)
    assert result["standard_call"] == "BA1_Strong"
    assert result["african_call"] == "BA1_Strong"
    assert result["flag"] is None


def test_classify_variant_with_african_context():
    """Test classify_variant with African context flag."""
    row = pd.Series({
        "variant_type": "SNP",
        "ref": "A", "alt": "T",
        "annotation": "missense_variant",
        "gnomad_af": 0.0001,
        "gnomad_af_afr": 0.015,  # African AF is much higher
        "ClinVar Significance": "",
        "info_raw": "",
    })
    result = classify_variant(row)
    assert result["acmg_class"] in ("VUS", "Likely Benign")  # African context downgrades PM2
    assert "african_context_flag" in result
    assert result["african_context_flag"] == "AFRICAN_CONTEXT_MISMATCH"
    assert "african_context_recommendation" in result

