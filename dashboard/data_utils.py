"""
Dashboard helper functions for data loading, KPI calculation, and recursive multi-step forecasting.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.config import (
    PROCESSED_DATA_FILE,
    FORECASTING_FEATURES_FILE,
    MODELS_DIR,
    REPORTS_DIR
)
from src.forecasting import get_feature_column_names, load_production_model


@st.cache_data
def load_transaction_data() -> pd.DataFrame:
    """Load cleaned transaction data with parsed dates."""
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Month"] = df["Order Date"].dt.month
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_forecasting_features_data() -> pd.DataFrame:
    """Load weekly forecasting feature dataset."""
    df = pd.read_csv(FORECASTING_FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource
def get_cached_model():
    """Load cached final production model pipeline."""
    return load_production_model()


@st.cache_data
def load_evaluation_reports() -> dict:
    """Load comparison CSVs and model metadata for display."""
    data = {}
    
    comp_file = REPORTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        data["comparison_df"] = pd.read_csv(comp_file)
        
    backtest_file = REPORTS_DIR / "backtesting_results.csv"
    if backtest_file.exists():
        data["backtest_df"] = pd.read_csv(backtest_file)
        
    worst_file = REPORTS_DIR / "worst_forecasts.csv"
    if worst_file.exists():
        data["worst_df"] = pd.read_csv(worst_file)
        
    meta_file = MODELS_DIR / "model_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)
            
    return data


def compute_executive_kpis(df: pd.DataFrame) -> dict:
    """Compute verified core executive KPIs from transactions DataFrame."""
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    profit_margin = (total_profit / total_sales * 100.0) if total_sales != 0 else 0.0
    total_units = int(df["Quantity"].sum())
    total_orders = int(df["Order ID"].nunique())
    aov = (total_sales / total_orders) if total_orders != 0 else 0.0
    
    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "profit_margin": profit_margin,
        "total_units": total_units,
        "total_orders": total_orders,
        "aov": aov
    }


def recursive_multistep_forecast(
    features_df: pd.DataFrame,
    model,
    horizon_weeks: int = 12
) -> pd.DataFrame:
    """
    Generate sequential recursive multi-step weekly forecasts for 1 to horizon_weeks.
    
    Methodology:
    - Step 1 uses actual historical lags up to 2017-12-31.
    - Steps 2..H update autoregressive sales lags using the model's own prior predictions.
    - Calendar features are calculated from the future calendar dates (W-SUN).
    - Operational driver lags (orders, quantity, discount, profit) use the 4-week trailing average.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Historical weekly feature dataset.
    model : Pipeline
        Fitted production model.
    horizon_weeks : int
        Number of weeks forward to project (1 to 12).
        
    Returns
    -------
    pd.DataFrame
        Forecasted dates, predicted sales, and lag-status indicators.
    """
    feature_cols = get_feature_column_names(features_df)
    last_date = features_df["Date"].max()
    
    # Maintain rolling sales buffer for dynamic lag updates
    sales_history = list(features_df["Sales"].values)
    
    # Trailing averages for operational features
    recent_4 = features_df.tail(4)
    avg_orders_lag = float(recent_4["orders_lag_1"].mean())
    avg_qty_lag = float(recent_4["quantity_lag_1"].mean())
    avg_prof_lag = float(recent_4["profit_lag_1"].mean())
    avg_disc_lag = float(recent_4["avg_discount_lag_1"].mean())
    
    forecast_records = []
    current_date = last_date
    
    for h in range(1, horizon_weeks + 1):
        next_date = current_date + pd.Timedelta(days=7)
        
        # 1. Calendar features
        year_val = next_date.year
        quarter_val = next_date.quarter
        month_val = next_date.month
        week_val = int(next_date.isocalendar().week)
        is_q4_val = 1 if quarter_val == 4 else 0
        sin_week_val = np.sin(2 * np.pi * week_val / 52.0)
        cos_week_val = np.cos(2 * np.pi * week_val / 52.0)
        sin_month_val = np.sin(2 * np.pi * month_val / 12.0)
        cos_month_val = np.cos(2 * np.pi * month_val / 12.0)
        
        # 2. Autoregressive lags from sales_history (sales_history[-1] is t-1, sales_history[-2] is t-2, etc.)
        s_lag_1 = sales_history[-1]
        s_lag_2 = sales_history[-2]
        s_lag_3 = sales_history[-3]
        s_lag_4 = sales_history[-4]
        s_lag_8 = sales_history[-8]
        s_lag_12 = sales_history[-12]
        s_lag_52 = sales_history[-52] if len(sales_history) >= 52 else s_lag_1
        
        # 3. Rolling window features
        rolling_4_window = sales_history[-4:]
        rolling_12_window = sales_history[-12:]
        
        s_roll_mean_4 = float(np.mean(rolling_4_window))
        s_roll_std_4 = float(np.std(rolling_4_window))
        s_roll_min_4 = float(np.min(rolling_4_window))
        s_roll_max_4 = float(np.max(rolling_4_window))
        s_roll_mean_12 = float(np.mean(rolling_12_window))
        s_roll_std_12 = float(np.std(rolling_12_window))
        
        # Construct single feature row
        row_dict = {
            "year": year_val,
            "quarter": quarter_val,
            "month": month_val,
            "week_of_year": week_val,
            "is_q4": is_q4_val,
            "sin_week": sin_week_val,
            "cos_week": cos_week_val,
            "sin_month": sin_month_val,
            "cos_month": cos_month_val,
            "sales_lag_1": s_lag_1,
            "sales_lag_2": s_lag_2,
            "sales_lag_3": s_lag_3,
            "sales_lag_4": s_lag_4,
            "sales_lag_8": s_lag_8,
            "sales_lag_12": s_lag_12,
            "sales_lag_52": s_lag_52,
            "sales_rolling_mean_4": s_roll_mean_4,
            "sales_rolling_std_4": s_roll_std_4,
            "sales_rolling_min_4": s_roll_min_4,
            "sales_rolling_max_4": s_roll_max_4,
            "sales_rolling_mean_12": s_roll_mean_12,
            "sales_rolling_std_12": s_roll_std_12,
            "orders_lag_1": avg_orders_lag,
            "quantity_lag_1": avg_qty_lag,
            "profit_lag_1": avg_prof_lag,
            "avg_discount_lag_1": avg_disc_lag
        }
        
        row_df = pd.DataFrame([row_dict])[feature_cols]
        step_pred = float(model.predict(row_df)[0])
        step_pred = max(0.0, step_pred)  # Non-negative sales guardrail
        
        forecast_records.append({
            "horizon_step": h,
            "forecast_date": next_date.strftime("%Y-%m-%d"),
            "predicted_sales": round(step_pred, 2),
            "lag_status": "Observed History" if h == 1 else f"Recursive ($h={h}$)"
        })
        
        # Append prediction to sales_history for the next iteration
        sales_history.append(step_pred)
        current_date = next_date
        
    return pd.DataFrame(forecast_records)
