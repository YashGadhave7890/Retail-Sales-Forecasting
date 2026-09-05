"""
Unit tests for src.feature_engineering and src.leakage_check modules.
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    aggregate_to_weekly,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    create_business_lags,
    build_forecasting_dataset
)
from src.leakage_check import audit_forecasting_leakage
from src.config import PROCESSED_DATA_FILE, FORECASTING_FEATURES_FILE


@pytest.fixture
def cleaned_df():
    """Load cleaned transaction dataset."""
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df


def test_chronological_ordering(cleaned_df):
    """Verify weekly aggregated series is strictly monotonically increasing with 7-day delta."""
    weekly = aggregate_to_weekly(cleaned_df)
    assert weekly.index.is_monotonic_increasing
    deltas = weekly.index.to_series().diff().dropna().dt.days
    assert (deltas == 7).all()


def test_lag_behavior(cleaned_df):
    """Verify lag features exactly match previous periods of the target variable."""
    feat_df, _ = build_forecasting_dataset(cleaned_df, drop_na=True)
    # Check lag 1
    assert np.isclose(
        feat_df["sales_lag_1"].iloc[1:].values,
        feat_df["Sales"].iloc[:-1].values,
        atol=1e-2
    ).all()
    # Check lag 2
    assert np.isclose(
        feat_df["sales_lag_2"].iloc[2:].values,
        feat_df["Sales"].iloc[:-2].values,
        atol=1e-2
    ).all()


def test_rolling_window_behavior(cleaned_df):
    """Verify rolling mean over 4 weeks strictly averages [t-4, t-1]."""
    feat_df, _ = build_forecasting_dataset(cleaned_df, drop_na=True)
    for i in range(4, 15):
        expected_mean = feat_df["Sales"].iloc[i-4 : i].mean()
        actual_mean = feat_df["sales_rolling_mean_4"].iloc[i]
        assert np.isclose(expected_mean, actual_mean, atol=1e-1)


def test_no_current_target_in_rolling(cleaned_df):
    """Verify current week's Sales is excluded from rolling window calculations."""
    feat_df, _ = build_forecasting_dataset(cleaned_df, drop_na=True)
    # If unshifted, rolling mean would be mean([i-3 : i+1])
    for i in range(4, 15):
        unshifted_mean = feat_df["Sales"].iloc[i-3 : i+1].mean()
        actual_mean = feat_df["sales_rolling_mean_4"].iloc[i]
        # Verify actual_mean does not equal unshifted_mean when sales[i] differs from sales[i-4]
        if not np.isclose(feat_df["Sales"].iloc[i], feat_df["Sales"].iloc[i-4], atol=1.0):
            assert not np.isclose(actual_mean, unshifted_mean, atol=1e-2)


def test_expected_feature_columns(cleaned_df):
    """Verify all expected predictor columns exist in the generated dataset."""
    feat_df, _ = build_forecasting_dataset(cleaned_df, drop_na=True)
    expected = [
        "Date", "Sales", "log_sales", "year", "quarter", "month", "week_of_year",
        "is_q4", "sin_week", "cos_week", "sin_month", "cos_month",
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_4",
        "sales_lag_8", "sales_lag_12", "sales_lag_52",
        "sales_rolling_mean_4", "sales_rolling_std_4", "sales_rolling_min_4", "sales_rolling_max_4",
        "sales_rolling_mean_12", "sales_rolling_std_12",
        "orders_lag_1", "quantity_lag_1", "profit_lag_1", "avg_discount_lag_1"
    ]
    for col in expected:
        assert col in feat_df.columns, f"Expected column '{col}' missing from forecasting dataset"


def test_no_contemporaneous_leakage_columns(cleaned_df):
    """Ensure raw unlagged operational columns (Profit, Quantity, Orders, Discount) are removed."""
    feat_df, _ = build_forecasting_dataset(cleaned_df, drop_na=True)
    forbidden = ["Profit", "Quantity", "Orders", "Avg_Discount", "Discount"]
    for col in forbidden:
        assert col not in feat_df.columns, f"Forbidden contemporaneous column '{col}' found in features!"


def test_deterministic_feature_generation(cleaned_df):
    """Verify that feature generation is 100% deterministic and reproducible."""
    df1, s1 = build_forecasting_dataset(cleaned_df, drop_na=True)
    df2, s2 = build_forecasting_dataset(cleaned_df, drop_na=True)
    pd.testing.assert_frame_equal(df1, df2)
    assert s1 == s2


def test_automated_leakage_audit_passes():
    """Verify that audit_forecasting_leakage confirms NO TARGET LEAKAGE DETECTED."""
    audit_results = audit_forecasting_leakage()
    assert audit_results["passed"] is True
    assert audit_results["overall_leakage_verdict"] == "NO TARGET LEAKAGE DETECTED"
