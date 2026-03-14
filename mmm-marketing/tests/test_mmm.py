"""Tests for mmm-marketing data loader."""
import io, sys, pathlib
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from data_loader import load_raw, validate

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
