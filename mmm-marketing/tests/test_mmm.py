"""Tests for mmm-marketing data loader."""
import io, sys, pathlib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from data_loader import load_raw, validate
from model import build_mmm
from sensitivity import channel_sensitivity, budget_reallocation

VALID_CSV = """\
date_week,y,x1,x2,event_1,event_2,dayofyear,t
2020-01-06,121213.4,213.5,312.9,0,0,6,1
2020-01-13,128413.2,198.7,340.1,0,0,13,2
2020-01-20,134211.0,221.1,298.6,1,0,20,3
"""

@pytest.fixture
def valid_df():
    return load_raw(io.StringIO(VALID_CSV))

class TestLoadRaw:
    def test_returns_dataframe(self, valid_df):
        assert isinstance(valid_df, pd.DataFrame)

    def test_date_column_is_datetime(self, valid_df):
        assert pd.api.types.is_datetime64_any_dtype(valid_df["date_week"])

    def test_row_count(self, valid_df):
        assert len(valid_df) == 3

    def test_accepts_file_like_object(self):
        df = load_raw(io.StringIO(VALID_CSV))
        assert len(df) == 3

class TestValidateHappyPath:
    def test_valid_df_returns_true(self, valid_df):
        ok, errors = validate(valid_df)
        assert ok is True and errors == []

class TestValidateFailureModes:
    def test_missing_required_column(self, valid_df):
        ok, errors = validate(valid_df.drop(columns=["y"]))
        assert ok is False
        assert any("y" in e for e in errors)

    def test_null_in_target(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "y"] = None
        ok, errors = validate(df)
        assert ok is False
        assert any("null" in e.lower() or "missing" in e.lower() for e in errors)

    def test_non_monotonic_dates(self):
        csv = "date_week,y,x1,x2,event_1,event_2,dayofyear,t\n2020-01-20,1,1,1,0,0,20,2\n2020-01-13,2,2,2,0,0,13,1\n"
        df = load_raw(io.StringIO(csv))
        ok, errors = validate(df)
        assert ok is False
        assert any("monoton" in e.lower() or "date" in e.lower() for e in errors)

    def test_null_in_spend_channel(self, valid_df):
        df = valid_df.copy()
        df.loc[1, "x1"] = None
        ok, errors = validate(df)
        assert ok is False


# ---------------------------------------------------------------------------
# Phase 3 Tests
# ---------------------------------------------------------------------------

class TestBuildMmm:
    def test_returns_mmm_instance(self):
        from pymc_marketing.mmm.multidimensional import MMM
        assert isinstance(build_mmm(), MMM)

    def test_default_channel_columns(self):
        assert build_mmm().channel_columns == ["x1", "x2"]

    def test_default_date_column(self):
        assert build_mmm().date_column == "date_week"

    def test_custom_channels(self):
        mmm = build_mmm(channel_columns=["x1"])
        assert mmm.channel_columns == ["x1"]


@pytest.fixture
def mock_mmm():
    """Minimal fitted MMM mock — avoids expensive MCMC in tests."""
    m = MagicMock()
    m.channel_columns = ["x1", "x2"]
    # Simulate posterior["channel_contribution"].values: (2 chains, 10 draws, 3 dates, 2 channels)
    contrib = MagicMock()
    contrib.values = np.ones((2, 10, 3, 2)) * 5.0
    m.idata.posterior.__getitem__.return_value = contrib
    return m


class TestChannelSensitivity:
    def test_returns_dataframe(self, mock_mmm, valid_df):
        assert isinstance(channel_sensitivity(mock_mmm, valid_df), pd.DataFrame)

    def test_index_is_channels(self, mock_mmm, valid_df):
        result = channel_sensitivity(mock_mmm, valid_df)
        assert list(result.index) == ["x1", "x2"]

    def test_required_columns(self, mock_mmm, valid_df):
        result = channel_sensitivity(mock_mmm, valid_df)
        for col in ["mean_contribution", "std_contribution", "pct_contribution"]:
            assert col in result.columns

    def test_pct_sums_to_100(self, mock_mmm, valid_df):
        result = channel_sensitivity(mock_mmm, valid_df)
        assert abs(result["pct_contribution"].sum() - 100.0) < 1e-6


class TestBudgetReallocation:
    def test_returns_dataframe(self, mock_mmm, valid_df):
        assert isinstance(budget_reallocation(mock_mmm, valid_df), pd.DataFrame)

    def test_index_is_channels(self, mock_mmm, valid_df):
        result = budget_reallocation(mock_mmm, valid_df)
        assert list(result.index) == ["x1", "x2"]

    def test_required_columns(self, mock_mmm, valid_df):
        result = budget_reallocation(mock_mmm, valid_df)
        for col in ["actual_spend", "actual_pct", "optimal_pct", "reallocation_delta_pct"]:
            assert col in result.columns
