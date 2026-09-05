"""
Unit tests for src.forecasting module.

Validates:
1. Final production model file exists on disk.
2. Production model metadata file exists and complies with schema.
3. Production model pipeline can be successfully loaded and invoked.
4. Predictor feature list matches expected 26 features.
5. Forecasting outputs are numeric and strictly finite.
6. Forecast DataFrame conforms to the expected schema (date, predicted_sales).
7. Example forecast export is valid and accessible.
8. Forecasting functions do not alter source raw, cleaned, or feature CSVs.
"""
import hashlib
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    FORECASTING_FEATURES_FILE,
    MODELS_DIR,
    REPORTS_DIR
)
from src.modeling import load_forecasting_data
from src.forecasting import (
    FINAL_MODEL_FILE,
    MODEL_METADATA_FILE,
    EXAMPLE_FORECAST_FILE,
    load_production_model,
    get_feature_column_names,
    generate_forecast
)

EXPECTED_RAW_SHA256 = "c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce"


def test_final_model_artifact_exists():
    """Verify that models/final_model.joblib exists and is non-empty."""
    assert FINAL_MODEL_FILE.exists(), f"Production model not found at {FINAL_MODEL_FILE}"
    assert FINAL_MODEL_FILE.stat().st_size > 500, "Production model file is unexpectedly small or empty"


def test_model_metadata_exists_and_valid():
    """Verify that models/model_metadata.json exists and conforms to required schema."""
    assert MODEL_METADATA_FILE.exists(), f"Model metadata not found at {MODEL_METADATA_FILE}"
    
    with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    required_keys = [
        "model_name",
        "model_artifact",
        "training_data_period",
        "total_training_observations",
        "target_variable",
        "frequency",
        "forecast_horizon",
        "feature_list",
        "feature_count",
        "scikit_learn_version",
        "unbiased_holdout_evaluation_metrics_2017",
        "known_limitations"
    ]
    for key in required_keys:
        assert key in metadata, f"Metadata missing required key: '{key}'"
        
    assert metadata["total_training_observations"] == 157
    assert metadata["feature_count"] == 26
    assert len(metadata["feature_list"]) == 26


def test_load_production_model():
    """Verify production model can be loaded and is an instance of sklearn.pipeline.Pipeline."""
    model = load_production_model()
    assert isinstance(model, Pipeline), "Loaded model is not an sklearn Pipeline"
    assert "scaler" in model.named_steps
    assert "reg" in model.named_steps


def test_expected_feature_list():
    """Verify get_feature_column_names extracts exactly the 26 expected features."""
    df = load_forecasting_data()
    feature_names = get_feature_column_names(df)
    assert len(feature_names) == 26
    assert "Sales" not in feature_names
    assert "log_sales" not in feature_names
    assert "Date" not in feature_names
    assert "sales_lag_1" in feature_names
    assert "sales_lag_52" in feature_names


def test_prediction_output_numeric_and_finite():
    """Verify forecast predictions are finite numeric values with no NaN or Inf."""
    df = load_forecasting_data()
    sample_features = df.head(10)
    forecast_df = generate_forecast(sample_features)
    
    assert len(forecast_df) == 10
    assert np.issubdtype(forecast_df["predicted_sales"].dtype, np.number)
    assert np.isfinite(forecast_df["predicted_sales"]).all()


def test_forecast_output_schema():
    """Verify output DataFrame schema is strictly ['date', 'predicted_sales']."""
    df = load_forecasting_data()
    forecast_df = generate_forecast(df.tail(15))
    
    assert list(forecast_df.columns) == ["date", "predicted_sales"]
    assert len(forecast_df) == 15
    # Verify date format YYYY-MM-DD
    pd.to_datetime(forecast_df["date"])


def test_example_forecast_csv_exists_and_valid():
    """Verify reports/example_forecast.csv exists and contains valid rows."""
    assert EXAMPLE_FORECAST_FILE.exists(), f"Example forecast CSV not found at {EXAMPLE_FORECAST_FILE}"
    df = pd.read_csv(EXAMPLE_FORECAST_FILE)
    assert list(df.columns) == ["date", "predicted_sales"]
    assert len(df) == 53
    assert np.isfinite(df["predicted_sales"]).all()


def test_forecasting_does_not_modify_source_data():
    """Verify forecasting functions do not modify raw, cleaned, or feature CSVs."""
    hasher = hashlib.sha256()
    with open(RAW_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    assert actual_hash == EXPECTED_RAW_SHA256, "Raw CSV was modified by forecasting module!"
    
    cleaned_df = pd.read_csv(PROCESSED_DATA_FILE)
    assert len(cleaned_df) == 9993, "Cleaned CSV row count modified!"
    
    feat_df = pd.read_csv(FORECASTING_FEATURES_FILE)
    assert len(feat_df) == 157, "Forecasting features row count modified!"
