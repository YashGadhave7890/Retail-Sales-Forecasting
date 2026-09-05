"""
Feature Engineering Module for Retail Sales Forecasting & Analytics.

Transforms cleaned retail transaction records into a weekly time-series forecasting
dataset with calendar, autoregressive lag, rolling-window, and operational features.
Guarantees strict target-leakage prevention by applying historical shift operations.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import PROCESSED_DATA_FILE, FORECASTING_FEATURES_FILE
from src.data_loader import load_raw_data

DEFAULT_LAGS: List[int] = [1, 2, 3, 4, 8, 12, 52]


def aggregate_to_weekly(df: pd.DataFrame, date_col: str = "Order Date") -> pd.DataFrame:
    """
    Aggregate transaction-level records into weekly Sunday-ended time series.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.
    date_col : str, default='Order Date'
        Name of date column.

    Returns
    -------
    pd.DataFrame
        Weekly aggregated DataFrame with continuous calendar index.
    """
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    df_copy = df_copy.set_index(date_col).sort_index()

    weekly = df_copy.resample("W-SUN").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Order ID": "nunique",
        "Discount": "mean"
    }).rename(columns={
        "Order ID": "Orders",
        "Discount": "Avg_Discount"
    })

    # Fill any potential zero-activity gaps
    weekly["Sales"] = weekly["Sales"].fillna(0.0)
    weekly["Profit"] = weekly["Profit"].fillna(0.0)
    weekly["Quantity"] = weekly["Quantity"].fillna(0)
    weekly["Orders"] = weekly["Orders"].fillna(0)
    weekly["Avg_Discount"] = weekly["Avg_Discount"].fillna(0.0)

    return weekly


def create_calendar_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate calendar and cyclical trigonometric features from the DatetimeIndex.
    All calendar features are deterministic and leakage-free.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with calendar features.
    """
    feat_df = weekly_df.copy()
    dates = feat_df.index

    feat_df["year"] = dates.year
    feat_df["quarter"] = dates.quarter
    feat_df["month"] = dates.month
    feat_df["week_of_year"] = dates.isocalendar().week.astype(int)
    
    # Q4 Indicator based on verified EDA seasonal findings (Sept-Dec represents 51.6% of sales)
    feat_df["is_q4"] = np.where(dates.month >= 9, 1, 0)

    # Cyclical trigonometric transforms for continuous periodicity
    feat_df["sin_week"] = np.sin(2 * np.pi * feat_df["week_of_year"] / 52.1775).round(4)
    feat_df["cos_week"] = np.cos(2 * np.pi * feat_df["week_of_year"] / 52.1775).round(4)
    feat_df["sin_month"] = np.sin(2 * np.pi * feat_df["month"] / 12.0).round(4)
    feat_df["cos_month"] = np.cos(2 * np.pi * feat_df["month"] / 12.0).round(4)

    return feat_df


def create_lag_features(
    weekly_df: pd.DataFrame,
    lags: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Generate strictly shifted historical lag features of the target variable.

    Rule: Target value at week t must NEVER enter its own feature vector.
    All lags use .shift(k) where k >= 1.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly DataFrame with 'Sales' column.
    lags : Optional[List[int]], default=None
        List of integer lag steps. Defaults to [1, 2, 3, 4, 8, 12, 52].

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with lag features.
    """
    feat_df = weekly_df.copy()
    target_lags = lags if lags is not None else DEFAULT_LAGS

    for k in target_lags:
        feat_df[f"sales_lag_{k}"] = feat_df["Sales"].shift(k)

    return feat_df


def create_rolling_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate rolling-window statistics computed strictly over historical observations.

    Rule: Must apply .shift(1) BEFORE .rolling(W) to guarantee that sales at week t
    is excluded from the rolling window.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly DataFrame with 'Sales' column.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with historical rolling statistics.
    """
    feat_df = weekly_df.copy()

    # Base historical series shifted by 1 week
    hist_sales = feat_df["Sales"].shift(1)

    # 4-week window (recent 1-month momentum)
    feat_df["sales_rolling_mean_4"] = hist_sales.rolling(4).mean().round(2)
    feat_df["sales_rolling_std_4"] = hist_sales.rolling(4).std().round(2)
    feat_df["sales_rolling_min_4"] = hist_sales.rolling(4).min().round(2)
    feat_df["sales_rolling_max_4"] = hist_sales.rolling(4).max().round(2)

    # 12-week window (quarterly baseline trend)
    feat_df["sales_rolling_mean_12"] = hist_sales.rolling(12).mean().round(2)
    feat_df["sales_rolling_std_12"] = hist_sales.rolling(12).std().round(2)

    return feat_df


def create_business_lags(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate strictly shifted historical lags for operational retail metrics.

    Rule: Contemporaneous values of Profit, Quantity, Orders, Discount at week t
    cannot be known at forecast time and are strictly excluded. Only week t-1 values
    are preserved as features.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Weekly DataFrame with operational columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with operational lag features and contemporaneous columns removed.
    """
    feat_df = weekly_df.copy()

    # Shift operational metrics by 1 week
    feat_df["orders_lag_1"] = feat_df["Orders"].shift(1)
    feat_df["quantity_lag_1"] = feat_df["Quantity"].shift(1)
    feat_df["profit_lag_1"] = feat_df["Profit"].shift(1).round(2)
    feat_df["avg_discount_lag_1"] = feat_df["Avg_Discount"].shift(1).round(4)

    # Drop contemporaneous operational columns to prevent leakage
    cols_to_drop = ["Profit", "Quantity", "Orders", "Avg_Discount"]
    feat_df = feat_df.drop(columns=[c for c in cols_to_drop if c in feat_df.columns])

    return feat_df


def build_forecasting_dataset(
    df: pd.DataFrame,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute full feature engineering pipeline on cleaned transaction records.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned retail transaction DataFrame.
    drop_na : bool, default=True
        Whether to drop initial rows with NaN values resulting from 52-week lag initialization.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (Forecasting dataset DataFrame, metadata summary dictionary).
    """
    # 1. Weekly aggregation
    weekly = aggregate_to_weekly(df)
    total_raw_weeks = len(weekly)

    # 2. Calendar features
    with_calendar = create_calendar_features(weekly)

    # 3. Lag features
    with_lags = create_lag_features(with_calendar)

    # 4. Rolling features
    with_rolling = create_rolling_features(with_lags)

    # 5. Business operational lags
    with_biz_lags = create_business_lags(with_rolling)

    # 6. Reset index to include 'Date' column
    final_df = with_biz_lags.reset_index().rename(columns={"Order Date": "Date"})

    # Add log-transformed target for model evaluation
    final_df["log_sales"] = np.log1p(final_df["Sales"]).round(4)

    # 7. Handle initial lag NaNs
    if drop_na:
        final_df = final_df.dropna().reset_index(drop=True)

    summary = {
        "total_weeks_aggregated": total_raw_weeks,
        "final_rows": len(final_df),
        "total_features": len(final_df.columns) - 2,  # Excluding Date and target Sales
        "target_column": "Sales",
        "date_range": {
            "start": str(final_df["Date"].min().strftime("%Y-%m-%d")),
            "end": str(final_df["Date"].max().strftime("%Y-%m-%d")),
        },
        "drop_na_applied": drop_na,
        "rows_dropped_for_lags": total_raw_weeks - len(final_df)
    }

    return final_df, summary


def save_forecasting_dataset(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Save the forecasting feature dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset to persist.
    output_path : Optional[Path], default=None
        Target path. Defaults to `src.config.FORECASTING_FEATURES_FILE`.

    Returns
    -------
    Path
        Path to saved file.
    """
    target = Path(output_path) if output_path is not None else FORECASTING_FEATURES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8")
    return target


def run_feature_engineering_pipeline(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run end-to-end feature engineering pipeline and save the forecasting dataset.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (Forecasting dataset DataFrame, summary dictionary).
    """
    src_file = Path(input_path) if input_path is not None else PROCESSED_DATA_FILE
    df_cleaned = pd.read_csv(src_file)
    df_features, summary = build_forecasting_dataset(df_cleaned, drop_na=True)
    saved_path = save_forecasting_dataset(df_features, output_path=output_path)
    summary["saved_path"] = str(saved_path.as_posix())
    return df_features, summary


if __name__ == "__main__":
    df_feat, summary = run_feature_engineering_pipeline()
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Aggregated Weeks: {summary['total_weeks_aggregated']}")
    print(f"Final Usable Rows: {summary['final_rows']}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Target Column: {summary['target_column']}")
    print(f"Engineered Features Count: {summary['total_features']}")
    print(f"Saved Dataset: {summary['saved_path']}")
    print("=" * 60)
