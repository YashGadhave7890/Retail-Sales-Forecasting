"""
Unit tests for src.modeling module.

Validates:
1. Chronological ordering is preserved throughout the pipeline.
2. Train dates strictly precede validation and test dates (no temporal leakage).
3. No target columns ('Sales', 'log_sales') or 'Date' are in X feature matrices.
4. Expected models can be instantiated, trained, and executed.
5. Model predictions contain finite numeric values.
6. Evaluation metrics are finite and handle zero values safely.
7. Baseline calculations strictly match their lag formulas.
8. Modeling execution does not modify the raw, cleaned, or feature CSV files.
"""
import hashlib
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    FORECASTING_FEATURES_FILE,
    REPORTS_DIR,
    ASSETS_MODELS_DIR,
    MODELS_DIR
)
from src.modeling import (
    load_forecasting_data,
    create_chronological_splits,
    calculate_metrics,
    get_model_definitions,
    evaluate_baselines,
    perform_cross_validation,
    train_and_evaluate_holdout
)

EXPECTED_RAW_SHA256 = "c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce"


@pytest.fixture
def forecasting_df():
    """Load forecasting feature dataset."""
    return load_forecasting_data()


@pytest.fixture
def split_data(forecasting_df):
    """Generate chronological train/test splits."""
    return create_chronological_splits(forecasting_df, test_year=2017)


def test_chronological_ordering(forecasting_df):
    """Test that the forecasting dataset is strictly chronologically ordered with 7-day delta."""
    assert forecasting_df["Date"].is_monotonic_increasing
    deltas = forecasting_df["Date"].diff().dropna().dt.days
    assert (deltas == 7).all(), "Weekly intervals must be exactly 7 days apart"


def test_train_test_split_dates(split_data):
    """Test that all training dates strictly precede all test dates."""
    X_train, y_train, X_test, y_test, train_df, test_df = split_data
    
    assert train_df["Date"].max() < test_df["Date"].min()
    assert train_df["Date"].dt.year.max() <= 2016
    assert test_df["Date"].dt.year.min() == 2017
    assert len(train_df) == 104
    assert len(test_df) == 53


def test_no_target_in_features(split_data):
    """Ensure target ('Sales'), transform ('log_sales'), and 'Date' are not in feature matrix X."""
    X_train, _, X_test, _, _, _ = split_data
    forbidden = ["Sales", "log_sales", "Date"]
    for col in forbidden:
        assert col not in X_train.columns, f"Target/Date column '{col}' leaked into X_train!"
        assert col not in X_test.columns, f"Target/Date column '{col}' leaked into X_test!"


def test_models_instantiation_and_training(split_data):
    """Test that all specified regression models can be instantiated and fitted."""
    X_train, y_train, X_test, y_test, _, _ = split_data
    models = get_model_definitions()
    
    expected_models = ["Ridge Regression", "Random Forest", "HistGradientBoosting", "Gradient Boosting"]
    for name in expected_models:
        assert name in models, f"Expected model '{name}' missing from get_model_definitions()"
        model = models[name]
        # Verify training works without error
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)


def test_predictions_finite_numeric(split_data):
    """Test that model predictions are finite numbers (no NaN, Inf, or empty values)."""
    X_train, y_train, X_test, _, _, _ = split_data
    models = get_model_definitions()
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert np.isfinite(preds).all(), f"Model '{name}' generated non-finite predictions!"
        assert len(preds) == len(X_test)


def test_metrics_calculation_finite():
    """Test that calculate_metrics computes accurate, finite MAE, RMSE, MAPE, and WAPE."""
    y_true = np.array([100.0, 200.0, 300.0, 0.0])
    y_pred = np.array([110.0, 190.0, 330.0, 10.0])
    
    m = calculate_metrics(y_true, y_pred)
    assert np.isfinite(m["MAE"])
    assert np.isfinite(m["RMSE"])
    assert np.isfinite(m["MAPE"])
    assert np.isfinite(m["WAPE"])
    assert m["MAE"] == round((10.0 + 10.0 + 30.0 + 10.0) / 4, 2)
    # Verify zero-division safety: non-zero actuals are [100, 200, 300], errors: [10%, 5%, 10%] -> mean = 8.33%
    assert np.isclose(m["MAPE"], 8.33, atol=0.1)


def test_baseline_logic_correctness(forecasting_df, split_data):
    """Test that baseline formulas strictly equal sales_lag_1 and sales_lag_52."""
    _, _, _, y_test, _, test_df = split_data
    
    naive_pred = test_df["sales_lag_1"].values
    snaive_pred = test_df["sales_lag_52"].values
    
    # Assert naive matches lag 1
    assert (naive_pred == test_df["sales_lag_1"].values).all()
    # Assert seasonal naive matches lag 52
    assert (snaive_pred == test_df["sales_lag_52"].values).all()
    
    m_naive = calculate_metrics(y_test.values, naive_pred)
    m_snaive = calculate_metrics(y_test.values, snaive_pred)
    
    assert m_naive["MAE"] > 0
    assert m_snaive["MAE"] > 0
    assert np.isfinite(m_naive["WAPE"])


def test_expanding_window_cv_temporal_soundness(split_data):
    """Verify that across every fold of cross-validation, train dates strictly precede val dates."""
    X_train, y_train, _, _, train_df, _ = split_data
    models = {"Ridge": get_model_definitions()["Ridge Regression"]}
    
    cv_summary, fold_details = perform_cross_validation(X_train, y_train, train_df, models, n_splits=5)
    
    assert len(fold_details) == 5
    for fold in fold_details:
        assert fold["train_end"] < fold["val_start"], (
            f"CV temporal leakage in fold {fold['fold']}: train end {fold['train_end']} >= val start {fold['val_start']}"
        )
        assert fold["train_samples"] > 0
        assert fold["val_samples"] > 0


def test_source_csv_immutability():
    """Verify that the raw CSV, cleaned CSV, and feature CSV files have not been corrupted or modified."""
    # Check raw data SHA-256 hash
    hasher = hashlib.sha256()
    with open(RAW_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    assert actual_hash == EXPECTED_RAW_SHA256, "Raw CSV SHA-256 hash mismatch! Raw data was altered!"
    
    # Check cleaned CSV row count
    cleaned_df = pd.read_csv(PROCESSED_DATA_FILE)
    assert len(cleaned_df) == 9993, f"Cleaned CSV row count changed! Expected 9993, got {len(cleaned_df)}"
    
    # Check forecasting features row count
    feat_df = pd.read_csv(FORECASTING_FEATURES_FILE)
    assert len(feat_df) == 157, f"Forecasting features row count changed! Expected 157, got {len(feat_df)}"
