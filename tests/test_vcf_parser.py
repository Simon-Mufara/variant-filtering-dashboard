"""Unit tests for utils/vcf_parser.py"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.vcf_parser import load_vcf, classify_ts_tv

EXAMPLE_VCF = os.path.join(os.path.dirname(__file__), "..", "data", "example.vcf")


class TestLoadVcf:
    def test_returns_dataframe(self):
        df = load_vcf(EXAMPLE_VCF)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = load_vcf(EXAMPLE_VCF)
        for col in ["chrom", "position", "ref", "alt", "quality", "depth", "variant_type"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_not_empty(self):
        df = load_vcf(EXAMPLE_VCF)
        assert len(df) > 0

    def test_af_column_present(self):
        df = load_vcf(EXAMPLE_VCF)
        assert "af" in df.columns

    def test_af_is_numeric(self):
        df = load_vcf(EXAMPLE_VCF)
        assert pd.api.types.is_float_dtype(df["af"])

    def test_variant_type_values(self):
        df = load_vcf(EXAMPLE_VCF)
        assert set(df["variant_type"].unique()).issubset({"SNP", "INDEL"})

    def test_multi_sample_columns(self):
        df = load_vcf(EXAMPLE_VCF)
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) >= 2, "Expected at least 2 sample columns"

    def test_quality_is_numeric(self):
        df = load_vcf(EXAMPLE_VCF)
        assert pd.api.types.is_float_dtype(df["quality"]) or pd.api.types.is_integer_dtype(df["quality"])

    def test_depth_is_numeric(self):
        df = load_vcf(EXAMPLE_VCF)
        assert pd.api.types.is_numeric_dtype(df["depth"])


class TestClassifyTsTv:
    def test_transition_a_to_g(self):
        assert classify_ts_tv("A", "G") == "Ts"

    def test_transition_c_to_t(self):
        assert classify_ts_tv("C", "T") == "Ts"

    def test_transversion_a_to_c(self):
        assert classify_ts_tv("A", "C") == "Tv"

    def test_transversion_g_to_c(self):
        assert classify_ts_tv("G", "C") == "Tv"

    def test_case_insensitive(self):
        assert classify_ts_tv("a", "g") == "Ts"
