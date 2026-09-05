"""
Unit tests for src.data_cleaning module.
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import (
    trim_text_columns,
    standardize_data_types,
    deduplicate_records,
    validate_categorical_columns,
    validate_numerical_columns,
    validate_date_chronology,
    clean_data,
    run_cleaning_pipeline
)
from src.validate_data import compute_file_hash
from src.config import RAW_DATA_FILE, PROCESSED_DATA_FILE

EXPECTED_SHA256: str = "c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce"


def test_string_normalization():
    """Verify that string trimming strips leading/trailing whitespace from all text columns."""
    sample_df = pd.DataFrame({
        "Product Name": ["  Mouse  ", "Keyboard  ", "Normal"],
        "Category": [" Tech ", "Tech", "Tech "]
    })
    cleaned = trim_text_columns(sample_df)
    assert cleaned["Product Name"].tolist() == ["Mouse", "Keyboard", "Normal"]
    assert cleaned["Category"].tolist() == ["Tech", "Tech", "Tech"]


def test_date_validation_success_and_failure():
    """Verify date chronology check passes on valid sequence and raises ValueError on invalid sequence."""
    valid_df = pd.DataFrame({
        "Order Date": pd.to_datetime(["2015-01-01", "2015-01-05"]),
        "Ship Date": pd.to_datetime(["2015-01-03", "2015-01-05"])
    })
    # Should pass without error
    validate_date_chronology(valid_df)

    invalid_df = pd.DataFrame({
        "Order Date": pd.to_datetime(["2015-01-05"]),
        "Ship Date": pd.to_datetime(["2015-01-01"])
    })
    with pytest.raises(ValueError, match="Order Date occurs after Ship Date"):
        validate_date_chronology(invalid_df)


def test_numerical_validation_bounds():
    """Verify numerical validation detects out-of-bound sales, quantities, and discounts."""
    valid_df = pd.DataFrame({
        "Sales": [10.5, 200.0],
        "Quantity": [1, 5],
        "Discount": [0.0, 0.5],
        "Profit": [-15.0, 50.0]  # Negative profit is valid
    })
    validate_numerical_columns(valid_df)

    # Test negative sales
    with pytest.raises(ValueError, match="non-positive values in 'Sales'"):
        validate_numerical_columns(pd.DataFrame({"Sales": [-5.0]}))

    # Test zero quantity
    with pytest.raises(ValueError, match="non-positive values in 'Quantity'"):
        validate_numerical_columns(pd.DataFrame({"Quantity": [0]}))

    # Test invalid discount > 1.0
    with pytest.raises(ValueError, match="out-of-bounds values in 'Discount'"):
        validate_numerical_columns(pd.DataFrame({"Discount": [1.5]}))


def test_duplicate_handling():
    """Verify deduplicate_records removes exact clones while preserving legitimate split lines."""
    df_cleaned, summary = clean_data()
    assert summary["raw_rows"] == 9994
    assert summary["cleaned_rows"] == 9993
    assert summary["rows_removed"] == 1

    # Ensure Row ID 3406 is kept and Row ID 3407 was removed
    assert 3406 in df_cleaned["Row ID"].values
    assert 3407 not in df_cleaned["Row ID"].values

    # Ensure legitimate multi-item orders are preserved
    split_order_df = df_cleaned[df_cleaned["Order ID"] == "CA-2015-103135"]
    assert len(split_order_df) >= 2


def test_cleaned_dataset_schema():
    """Verify the processed dataset maintains expected schema, column count, and standardized dtypes."""
    df_cleaned, _ = clean_data()
    assert df_cleaned.shape == (9993, 21)
    assert "Postal Code" in df_cleaned.columns
    # Check postal code is 5-digit string
    assert df_cleaned["Postal Code"].astype(str).str.len().eq(5).all()
    # Check datetime types
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned["Order Date"])
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned["Ship Date"])


def test_no_unexpected_null_values():
    """Ensure that the cleaned dataset has zero missing values."""
    df_cleaned, _ = clean_data()
    assert df_cleaned.isnull().sum().sum() == 0


def test_raw_dataset_immutability():
    """Verify that running the cleaning pipeline has not altered the raw CSV hash."""
    current_hash = compute_file_hash(RAW_DATA_FILE)
    assert current_hash == EXPECTED_SHA256, (
        f"Raw file was modified! Expected hash {EXPECTED_SHA256}, got {current_hash}"
    )


def test_cleaning_pipeline_reproducible():
    """Verify that running the cleaning pipeline multiple times produces identical output."""
    df1, summary1 = clean_data()
    df2, summary2 = clean_data()
    pd.testing.assert_frame_equal(df1, df2)
    assert summary1 == summary2
