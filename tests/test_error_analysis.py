"""
Unit tests for src.error_analysis module.

Validates:
1. Error analysis dataset schema and required columns.
2. Residual and absolute error mathematical correctness (e = y - y_hat).
3. Observation count strictly matches the 53 test weeks of 2017.
4. Objective quartile sales tier partitioning is exhaustive and mutually exclusive.
5. Worst forecast extraction is properly sorted and accurately classified.
6. Statistical diagnostic test values are finite and mathematically valid.
7. Raw and cleaned datasets remain completely unaltered.
"""
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    FORECASTING_FEATURES_FILE
)
from src.modeling import (
    load_forecasting_data,
    create_chronological_splits
)
from src.error_analysis import (
    compute_error_analysis_dataset,
    get_worst_forecasts
)

EXPECTED_RAW_SHA256 = "c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce"


@pytest.fixture(scope="module")
def error_data():
    """Compute error analysis once for test suite."""
    df = load_forecasting_data()
    X_train, y_train, X_test, y_test, train_df, test_df = create_chronological_splits(df, test_year=2017)
    err_df, stats_dict = compute_error_analysis_dataset(X_train, y_train, X_test, y_test, test_df, model_name="Ridge Regression")
    worst_df = get_worst_forecasts(err_df, top_n=10)
    return err_df, stats_dict, worst_df, test_df


def test_error_analysis_dataset_columns(error_data):
    """Test that the error analysis DataFrame contains all required fields."""
    err_df, _, _, _ = error_data
    expected_cols = [
        "Date", "Actual", "Predicted", "Residual", "Absolute_Error",
        "APE (%)", "Month", "Quarter", "Week", "Sales_Tier"
    ]
    for col in expected_cols:
        assert col in err_df.columns, f"Expected column '{col}' missing from error dataset!"


def test_residual_math_invariant(error_data):
    """Test that Residual = Actual - Predicted and Absolute_Error = |Residual|."""
    err_df, _, _, _ = error_data
    calc_res = np.round(err_df["Actual"] - err_df["Predicted"], 2)
    assert np.allclose(err_df["Residual"], calc_res, atol=1e-2), "Residual calculation mismatch!"
    
    calc_abs = np.round(np.abs(err_df["Residual"]), 2)
    assert np.allclose(err_df["Absolute_Error"], calc_abs, atol=1e-2), "Absolute error calculation mismatch!"


def test_observation_count_matches_2017_holdout(error_data):
    """Test that error dataset contains exactly 53 weekly observations (all 2017 weeks)."""
    err_df, _, _, test_df = error_data
    assert len(err_df) == 53
    assert len(test_df) == 53
    assert err_df["Date"].dt.year.unique().tolist() == [2017]


def test_tier_partition_covers_all_samples(error_data):
    """Test that sales tiers are exhaustive and match quartile distribution."""
    err_df, stats_dict, _, _ = error_data
    tier_counts = err_df["Sales_Tier"].value_counts()
    
    assert set(tier_counts.index) == {
        "Low (<25th pct)",
        "Normal (25th-75th pct)",
        "High (>75th pct)"
    }
    assert tier_counts.sum() == 53
    # Check that high tier values exceed q75 and low tier values are below q25
    low_actuals = err_df[err_df["Sales_Tier"] == "Low (<25th pct)"]["Actual"]
    high_actuals = err_df[err_df["Sales_Tier"] == "High (>75th pct)"]["Actual"]
    assert (low_actuals < stats_dict["q25_threshold"]).all()
    assert (high_actuals > stats_dict["q75_threshold"]).all()


def test_worst_forecasts_sorting_and_direction(error_data):
    """Test that worst forecasts are properly sorted and categorized by error direction."""
    _, _, worst_df, _ = error_data
    assert len(worst_df) == 10
    
    # Check strictly non-increasing order of absolute error
    abs_errors = worst_df["Absolute_Error"].tolist()
    assert all(abs_errors[i] >= abs_errors[i+1] for i in range(len(abs_errors)-1))
    
    # Check error direction logic
    for _, row in worst_df.iterrows():
        if row["Residual"] > 0:
            assert "Underpredicted" in row["Error_Direction"]
        else:
            assert "Overpredicted" in row["Error_Direction"]


def test_diagnostic_statistics_finite(error_data):
    """Test that all statistical diagnostics produce valid, finite values."""
    _, stats_dict, _, _ = error_data
    assert np.isfinite(stats_dict["t_test_stat"])
    assert 0.0 <= stats_dict["t_test_pval"] <= 1.0
    assert np.isfinite(stats_dict["shapiro_stat"])
    assert 0.0 <= stats_dict["shapiro_pval"] <= 1.0
    assert 0.0 <= stats_dict["durbin_watson"] <= 4.0


def test_raw_and_cleaned_immutability():
    """Verify raw CSV SHA-256 hash and processed row counts remain untouched."""
    hasher = hashlib.sha256()
    with open(RAW_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    assert actual_hash == EXPECTED_RAW_SHA256, "Raw CSV SHA-256 hash modified!"
    
    cleaned = pd.read_csv(PROCESSED_DATA_FILE)
    assert len(cleaned) == 9993, f"Cleaned CSV row count modified! ({len(cleaned)})"
    
    feat = pd.read_csv(FORECASTING_FEATURES_FILE)
    assert len(feat) == 157, f"Forecasting features row count modified! ({len(feat)})"
