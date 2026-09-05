"""
Professional Data Cleaning and Processed Dataset Creation Module.

Provides modular, testable, business-aware data cleaning functions for the
Superstore retail transactions dataset without fabricating or arbitrarily
removing records.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import RAW_DATA_FILE, PROCESSED_DATA_FILE
from src.data_loader import load_raw_data

EXPECTED_CATEGORIES: List[str] = ["Furniture", "Office Supplies", "Technology"]
EXPECTED_SEGMENTS: List[str] = ["Consumer", "Corporate", "Home Office"]
EXPECTED_REGIONS: List[str] = ["Central", "East", "South", "West"]
EXPECTED_SHIP_MODES: List[str] = ["First Class", "Same Day", "Second Class", "Standard Class"]

STRING_COLUMNS_TO_TRIM: List[str] = [
    "Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment",
    "Country", "City", "State", "Region", "Product ID", "Category",
    "Sub-Category", "Product Name"
]


def trim_text_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Strip leading and trailing whitespace from string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Optional[List[str]], default=None
        List of column names to trim. If None, uses default list of text columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with trimmed text columns.
    """
    df_clean = df.copy()
    target_cols = columns if columns is not None else STRING_COLUMNS_TO_TRIM

    for col in target_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean


def standardize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column data types to appropriate pandas representation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized data types.
    """
    df_clean = df.copy()

    # Ensure dates are datetime64
    for date_col in ["Order Date", "Ship Date"]:
        if date_col in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean[date_col]):
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], format="mixed", dayfirst=False)

    # Postal code formatted as 5-digit zero-padded string (geospatial categorical best practice)
    if "Postal Code" in df_clean.columns:
        df_clean["Postal Code"] = df_clean["Postal Code"].astype(str).str.split(".").str[0].str.zfill(5)

    # Numeric columns
    if "Sales" in df_clean.columns:
        df_clean["Sales"] = df_clean["Sales"].astype(float)
    if "Quantity" in df_clean.columns:
        df_clean["Quantity"] = df_clean["Quantity"].astype(int)
    if "Discount" in df_clean.columns:
        df_clean["Discount"] = df_clean["Discount"].astype(float)
    if "Profit" in df_clean.columns:
        df_clean["Profit"] = df_clean["Profit"].astype(float)

    return df_clean


def deduplicate_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identify and remove confirmed accidental double-entry duplicate records.

    Business Rationale:
    When excluding surrogate `Row ID`, records that are 100% identical across all
    20 business attributes represent accidental logging duplicates. Multi-item
    order lines with differing quantities/sales are legitimate split orders and
    are strictly preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        (Cleaned DataFrame, number of duplicate records removed).
    """
    df_clean = df.copy()
    business_cols = [col for col in df_clean.columns if col != "Row ID"]

    # Detect duplicate rows across all business attributes
    dup_mask = df_clean.duplicated(subset=business_cols, keep="first")
    num_removed = int(dup_mask.sum())

    df_deduped = df_clean[~dup_mask].copy().reset_index(drop=True)
    return df_deduped, num_removed


def validate_categorical_columns(df: pd.DataFrame) -> None:
    """
    Validate that categorical columns contain only expected domain values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    -------
    ValueError
        If an unexpected category value is detected.
    """
    checks = [
        ("Category", EXPECTED_CATEGORIES),
        ("Segment", EXPECTED_SEGMENTS),
        ("Region", EXPECTED_REGIONS),
        ("Ship Mode", EXPECTED_SHIP_MODES),
    ]
    for col, expected in checks:
        if col in df.columns:
            actual = set(df[col].unique())
            unexpected = actual - set(expected)
            if unexpected:
                raise ValueError(
                    f"Unexpected categories in '{col}': {unexpected}. Expected: {expected}"
                )


def validate_numerical_columns(df: pd.DataFrame) -> None:
    """
    Validate numerical column bounds (Sales > 0, Quantity >= 1, 0 <= Discount <= 1).

    Note: Negative profit is NOT an error and is explicitly validated as valid
    business loss.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    -------
    ValueError
        If any numerical boundary invariant is violated.
    """
    if "Sales" in df.columns and (df["Sales"] <= 0).any():
        invalid_count = (df["Sales"] <= 0).sum()
        raise ValueError(f"Found {invalid_count} non-positive values in 'Sales'")

    if "Quantity" in df.columns and (df["Quantity"] <= 0).any():
        invalid_count = (df["Quantity"] <= 0).sum()
        raise ValueError(f"Found {invalid_count} non-positive values in 'Quantity'")

    if "Discount" in df.columns:
        out_of_bounds = ((df["Discount"] < 0) | (df["Discount"] > 1.0)).sum()
        if out_of_bounds > 0:
            raise ValueError(f"Found {out_of_bounds} out-of-bounds values in 'Discount'")


def validate_date_chronology(df: pd.DataFrame) -> None:
    """
    Validate that Order Date is strictly prior to or equal to Ship Date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    -------
    ValueError
        If any record has Order Date > Ship Date.
    """
    if "Order Date" in df.columns and "Ship Date" in df.columns:
        violations = (df["Order Date"] > df["Ship Date"]).sum()
        if violations > 0:
            raise ValueError(
                f"Found {violations} records where Order Date occurs after Ship Date"
            )


def clean_data(df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the full reproducible data-cleaning pipeline.

    Parameters
    ----------
    df : Optional[pd.DataFrame], default=None
        DataFrame to clean. If None, loads raw data using load_raw_data().

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (Cleaned DataFrame, metadata dictionary summarizing transformations).
    """
    df_raw = df.copy() if df is not None else load_raw_data(parse_dates=True)
    raw_rows, raw_cols = df_raw.shape

    # 1. Text Trimming
    df_trimmed = trim_text_columns(df_raw)

    # 2. Type Standardization
    df_standardized = standardize_data_types(df_trimmed)

    # 3. Deduplication of confirmed accidental duplicates
    df_deduped, rows_removed = deduplicate_records(df_standardized)

    # 4. Invariant Validation
    validate_categorical_columns(df_deduped)
    validate_numerical_columns(df_deduped)
    validate_date_chronology(df_deduped)

    cleaned_rows, cleaned_cols = df_deduped.shape

    summary = {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed": rows_removed,
        "rows_retained": cleaned_rows,
        "columns": cleaned_cols,
        "negative_profit_rows_retained": int((df_deduped["Profit"] < 0).sum()),
        "text_trimmed_columns": STRING_COLUMNS_TO_TRIM,
        "postal_code_standardized": True
    }

    return df_deduped, summary


def save_processed_data(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """
    Save the cleaned DataFrame to CSV format.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to persist.
    output_path : Optional[Path], default=None
        Target file path. Defaults to `src.config.PROCESSED_DATA_FILE`.

    Returns
    -------
    Path
        Path to the saved processed file.
    """
    target = Path(output_path) if output_path is not None else PROCESSED_DATA_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")
    return target


def run_cleaning_pipeline(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run end-to-end data cleaning and persist the processed dataset.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (Cleaned DataFrame, summary dictionary).
    """
    df_raw = load_raw_data(file_path=raw_path, parse_dates=True)
    df_cleaned, summary = clean_data(df_raw)
    saved_path = save_processed_data(df_cleaned, output_path=output_path)
    summary["saved_path"] = str(saved_path.as_posix())
    return df_cleaned, summary


if __name__ == "__main__":
    df_cleaned, summary = run_cleaning_pipeline()
    print("=" * 60)
    print("DATA CLEANING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Raw rows: {summary['raw_rows']}")
    print(f"Cleaned rows: {summary['cleaned_rows']}")
    print(f"Rows removed: {summary['rows_removed']}")
    print(f"Rows retained: {summary['rows_retained']}")
    print(f"Columns: {summary['columns']}")
    print(f"Negative-profit rows preserved: {summary['negative_profit_rows_retained']}")
    print(f"Saved processed dataset: {summary['saved_path']}")
    print("=" * 60)
