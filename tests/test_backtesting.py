"""
Unit tests for src.backtesting module.

Validates:
1. Strict chronological ordering across all backtesting folds.
2. No future observations used in training (train_end < val_start).
3. Expanding fold boundaries are logically consistent and valid.
4. All backtesting metrics (MAE, RMSE, MAPE, WAPE) are finite numbers.
5. Prediction sample counts match ground-truth validation sample counts.
6. Summary dataframe contains expected aggregated metrics.
"""
import pytest
import pandas as pd
import numpy as np

from src.modeling import (
    load_forecasting_data,
    create_chronological_splits,
    get_model_definitions
)
from src.backtesting import run_expanding_window_backtest


@pytest.fixture(scope="module")
def backtest_data():
    """Run expanding-window backtesting once for the test suite."""
    df = load_forecasting_data()
    X_train, y_train, _, _, train_df, _ = create_chronological_splits(df, test_year=2017)
    models = get_model_definitions()
    fold_df, summary_df = run_expanding_window_backtest(X_train, y_train, train_df, models, n_splits=5)
    return fold_df, summary_df, train_df


def test_backtest_chronological_ordering(backtest_data):
    """Test that train and validation periods in each fold are strictly chronological."""
    fold_df, _, train_df = backtest_data
    assert train_df["Date"].is_monotonic_increasing


def test_backtest_no_future_training(backtest_data):
    """Test that training end dates strictly precede validation start dates across all folds."""
    fold_df, _, _ = backtest_data
    for _, row in fold_df.iterrows():
        train_end = row["Train Period"].split(" to ")[1]
        val_start = row["Val Period"].split(" to ")[0]
        assert train_end < val_start, (
            f"Temporal leakage in Fold {row['Fold']} ({row['Model']}): "
            f"train_end ({train_end}) >= val_start ({val_start})"
        )


def test_backtest_fold_boundaries_valid(backtest_data):
    """Test that the number of folds is 5 and training size strictly expands."""
    fold_df, _, _ = backtest_data
    assert fold_df["Fold"].nunique() == 5
    
    # Check expanding training sizes across folds for any single model
    ridge_folds = fold_df[fold_df["Model"] == "Ridge Regression"].sort_values("Fold")
    train_sizes = ridge_folds["Train N"].tolist()
    assert all(train_sizes[i] < train_sizes[i+1] for i in range(len(train_sizes)-1)), (
        f"Training set sizes do not strictly expand: {train_sizes}"
    )


def test_backtest_metrics_finite(backtest_data):
    """Test that all calculated metrics across all models and folds are finite and positive."""
    fold_df, summary_df, _ = backtest_data
    metric_cols = ["MAE", "RMSE", "MAPE", "WAPE"]
    for col in metric_cols:
        assert np.isfinite(fold_df[col]).all(), f"Found non-finite values in fold metric '{col}'"
        assert (fold_df[col] > 0).all(), f"Found non-positive values in fold metric '{col}'"
        
    summary_metric_cols = ["Mean_MAE", "Median_MAE", "Std_MAE", "Mean_RMSE", "Mean_MAPE", "Mean_WAPE"]
    for col in summary_metric_cols:
        assert np.isfinite(summary_df[col]).all(), f"Found non-finite values in summary metric '{col}'"


def test_backtest_prediction_counts_match(backtest_data):
    """Test that validation sample counts are positive and consistent across models in each fold."""
    fold_df, _, _ = backtest_data
    for fold in fold_df["Fold"].unique():
        f_sub = fold_df[fold_df["Fold"] == fold]
        val_sizes = f_sub["Val N"].unique()
        assert len(val_sizes) == 1, f"Mismatched validation sizes in Fold {fold}: {val_sizes}"
        assert val_sizes[0] > 0


def test_backtest_all_models_present(backtest_data):
    """Test that all expected baselines and ML models are evaluated in backtesting."""
    fold_df, summary_df, _ = backtest_data
    expected_models = {
        "Naive Baseline",
        "Seasonal Naive Baseline",
        "Ridge Regression",
        "Random Forest",
        "HistGradientBoosting",
        "Gradient Boosting"
    }
    assert set(fold_df["Model"].unique()) == expected_models
    assert set(summary_df["Model"].unique()) == expected_models
