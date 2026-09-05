"""
Unit tests for Streamlit dashboard and helper utilities.

Validates:
1. Dashboard and data_utils modules import cleanly.
2. Required dashboard and model artifacts exist on disk.
3. Production model loads properly and can generate predictions.
4. Dashboard transaction and forecasting datasets load without error.
5. Required analytical and feature columns exist.
6. Executive KPI calculations return mathematically sound, finite values.
7. Recursive multi-step forecasting output schema is correct.
8. Dashboard execution does not alter raw or processed datasets.
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
    MODELS_DIR,
    PROJECT_ROOT
)
from dashboard.data_utils import (
    load_transaction_data,
    load_forecasting_features_data,
    get_cached_model,
    compute_executive_kpis,
    recursive_multistep_forecast,
    load_evaluation_reports
)

EXPECTED_RAW_SHA256 = "c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce"
DASHBOARD_APP_FILE = PROJECT_ROOT / "dashboard" / "app.py"
DASHBOARD_README_FILE = PROJECT_ROOT / "dashboard" / "README.md"
FINAL_MODEL_FILE = MODELS_DIR / "final_model.joblib"


def test_dashboard_imports():
    """Verify dashboard data utilities can be imported without error."""
    import dashboard.data_utils as du
    assert hasattr(du, "load_transaction_data")
    assert hasattr(du, "recursive_multistep_forecast")


def test_dashboard_required_files_exist():
    """Verify required dashboard application files and model artifacts exist."""
    assert DASHBOARD_APP_FILE.exists(), "dashboard/app.py missing!"
    assert DASHBOARD_README_FILE.exists(), "dashboard/README.md missing!"
    assert FINAL_MODEL_FILE.exists(), "models/final_model.joblib missing!"
    assert PROCESSED_DATA_FILE.exists(), "Cleaned dataset missing!"
    assert FORECASTING_FEATURES_FILE.exists(), "Forecasting features missing!"


def test_dashboard_model_loads():
    """Verify production model loads and possesses required pipeline steps."""
    model = get_cached_model()
    assert hasattr(model, "predict")
    assert "scaler" in model.named_steps
    assert "reg" in model.named_steps


def test_dashboard_data_loads():
    """Verify transaction and forecasting datasets load with correct record counts."""
    t_df = load_transaction_data()
    f_df = load_forecasting_features_data()
    
    assert len(t_df) == 9993, f"Expected 9,993 transactions, got {len(t_df)}"
    assert len(f_df) == 157, f"Expected 157 weekly feature rows, got {len(f_df)}"


def test_dashboard_required_columns():
    """Verify expected analytical columns exist in the loaded transaction DataFrame."""
    t_df = load_transaction_data()
    required_cols = [
        "Order Date", "Sales", "Profit", "Quantity", "Order ID",
        "Category", "Sub-Category", "Segment", "Region", "State",
        "Year", "Quarter", "Month", "YearMonth"
    ]
    for col in required_cols:
        assert col in t_df.columns, f"Required column '{col}' missing from transaction dataset!"


def test_kpi_calculations_finite_and_accurate():
    """Verify compute_executive_kpis returns finite values matching verified numbers."""
    t_df = load_transaction_data()
    kpis = compute_executive_kpis(t_df)
    
    assert np.isclose(kpis["total_sales"], 2296919.49, atol=1.0)
    assert np.isclose(kpis["total_profit"], 286409.08, atol=1.0)
    assert np.isclose(kpis["profit_margin"], 12.47, atol=0.1)
    assert kpis["total_units"] == 37871
    assert kpis["total_orders"] == 5009
    assert np.isclose(kpis["aov"], 458.56, atol=0.1)
    
    for k, v in kpis.items():
        assert np.isfinite(v), f"KPI '{k}' returned non-finite value: {v}"


def test_recursive_forecast_output_schema():
    """Verify recursive multi-step forecasting produces correct schema, horizon counts, and finite values."""
    f_df = load_forecasting_features_data()
    model = get_cached_model()
    
    for h in [1, 4, 12]:
        fc = recursive_multistep_forecast(f_df, model, horizon_weeks=h)
        assert len(fc) == h, f"Expected {h} rows, got {len(fc)}"
        assert list(fc.columns) == ["horizon_step", "forecast_date", "predicted_sales", "lag_status"]
        assert np.isfinite(fc["predicted_sales"]).all()
        assert (fc["predicted_sales"] >= 0).all(), "Found negative forecasted sales!"
        # Check Step 1 uses Observed History
        assert fc.iloc[0]["lag_status"] == "Observed History"
        if h > 1:
            assert "Recursive" in fc.iloc[1]["lag_status"]


def test_load_evaluation_reports():
    """Verify evaluation reports helper successfully loads comparison tables."""
    reports = load_evaluation_reports()
    assert "comparison_df" in reports
    assert "backtest_df" in reports
    assert "metadata" in reports
    assert len(reports["comparison_df"]) >= 6


def test_source_datasets_unmodified():
    """Verify running dashboard functions does not modify raw or cleaned CSVs."""
    hasher = hashlib.sha256()
    with open(RAW_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    assert actual_hash == EXPECTED_RAW_SHA256, "Raw CSV was altered!"
    
    cleaned_df = pd.read_csv(PROCESSED_DATA_FILE)
    assert len(cleaned_df) == 9993, "Cleaned CSV altered!"
    
    feat_df = pd.read_csv(FORECASTING_FEATURES_FILE)
    assert len(feat_df) == 157, "Forecasting features altered!"
