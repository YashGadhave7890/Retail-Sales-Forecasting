"""
Unit tests for src.data_loader module.
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, DEFAULT_DATE_COLUMNS
from src.config import RAW_DATA_FILE


def test_load_raw_data_success():
    """Verify that raw dataset loads successfully as a non-empty DataFrame with expected shape."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (9994, 21)


def test_expected_columns_exist():
    """Verify that all standard columns are present in the loaded DataFrame."""
    df = load_raw_data()
    expected_cols = [
        "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode",
        "Customer ID", "Customer Name", "Segment", "Country", "City",
        "State", "Postal Code", "Region", "Product ID", "Category",
        "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' missing from loaded dataset"


def test_date_parsing():
    """Verify that date columns are parsed as datetime64 when parse_dates=True."""
    df = load_raw_data(parse_dates=True)
    for col in DEFAULT_DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(df[col]), (
            f"Column '{col}' should be datetime64, got {df[col].dtype}"
        )


def test_unparsed_dates_preserved():
    """Verify that date columns remain strings when parse_dates=False."""
    df = load_raw_data(parse_dates=False)
    for col in DEFAULT_DATE_COLUMNS:
        assert not pd.api.types.is_datetime64_any_dtype(df[col]), (
            f"Column '{col}' should not be datetime when parse_dates=False"
        )
        assert pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object, (
            f"Column '{col}' should be string/object when parse_dates=False, got {df[col].dtype}"
        )


def test_missing_file_raises_error(tmp_path: Path):
    """Verify that attempting to load a non-existent file raises a descriptive FileNotFoundError."""
    non_existent = tmp_path / "non_existent_dataset.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset not found at"):
        load_raw_data(file_path=non_existent)


def test_no_row_omission():
    """Ensure that the data loader does not drop any records or perform silent truncation."""
    df = load_raw_data()
    assert len(df) == 9994
