"""Unit tests for utils/filters.py"""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.vcf_parser import load_vcf
from utils.filters import apply_filters

EXAMPLE_VCF = os.path.join(os.path.dirname(__file__), "..", "data", "example.vcf")


@pytest.fixture
def sample_df():
    return load_vcf(EXAMPLE_VCF)


class TestApplyFilters:
    def test_returns_dataframe(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="All")
        assert isinstance(result, pd.DataFrame)

    def test_min_quality_filters(self, sample_df):
        result = apply_filters(sample_df, min_quality=50, min_depth=0, variant_type="All")
        assert (result["quality"] >= 50).all()

    def test_min_depth_filters(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=30, variant_type="All")
        assert (result["depth"] >= 30).all()

    def test_variant_type_snp(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="SNP")
        assert (result["variant_type"] == "SNP").all()

    def test_variant_type_indel(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="INDEL")
        assert (result["variant_type"] == "INDEL").all()

    def test_variant_type_all(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="All")
        assert len(result) == len(sample_df)

    def test_chromosome_filter(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="All", chromosomes=["chr1"])
        assert (result["chrom"] == "chr1").all()

    def test_af_min_filter(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="All", min_af=0.4)
        valid = result.dropna(subset=["af"])
        assert (valid["af"] >= 0.4).all()

    def test_af_max_filter(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="All", max_af=0.3)
        valid = result.dropna(subset=["af"])
        assert (valid["af"] <= 0.3).all()

    def test_strict_filters_reduce_rows(self, sample_df):
        result = apply_filters(sample_df, min_quality=80, min_depth=50, variant_type="SNP")
        assert len(result) <= len(sample_df)

    def test_no_filters_returns_all(self, sample_df):
        result = apply_filters(sample_df, min_quality=0, min_depth=0, variant_type="All")
        assert len(result) == len(sample_df)

    def test_result_is_reset_index(self, sample_df):
        result = apply_filters(sample_df, min_quality=50, min_depth=0, variant_type="All")
        assert list(result.index) == list(range(len(result)))
