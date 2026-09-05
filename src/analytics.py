"""
Analytical and Business KPI Calculation Module for Retail Sales Forecasting & Analytics.

Provides reusable, tested functions for computing core business metrics,
temporal aggregations, and segment breakdowns from cleaned retail transaction data.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


def calculate_core_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate high-level retail business KPIs.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned retail transaction DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of core KPIs including total sales, total profit, total quantity,
        order count, customer count, product count, AOV, avg profit/order, and margin.
    """
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    total_quantity = int(df["Quantity"].sum())
    num_orders = int(df["Order ID"].nunique())
    num_customers = int(df["Customer ID"].nunique())
    num_products = int(df["Product ID"].nunique())

    # Order-level aggregations
    order_agg = df.groupby("Order ID").agg({"Sales": "sum", "Profit": "sum"})
    aov = float(order_agg["Sales"].mean()) if num_orders > 0 else 0.0
    avg_profit_per_order = float(order_agg["Profit"].mean()) if num_orders > 0 else 0.0
    overall_profit_margin = float((total_profit / total_sales) * 100) if total_sales > 0 else 0.0

    return {
        "total_sales": round(total_sales, 2),
        "total_profit": round(total_profit, 2),
        "total_quantity": total_quantity,
        "num_orders": num_orders,
        "num_customers": num_customers,
        "num_products": num_products,
        "average_order_value": round(aov, 2),
        "average_profit_per_order": round(avg_profit_per_order, 2),
        "overall_profit_margin_pct": round(overall_profit_margin, 2),
    }


def aggregate_by_time(
    df: pd.DataFrame,
    freq: str = "ME",
    date_col: str = "Order Date"
) -> pd.DataFrame:
    """
    Aggregate sales, profit, and order count at a given temporal frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a datetime date_col.
    freq : str, default='ME'
        Pandas offset alias ('D' for daily, 'W-SUN' for weekly, 'ME' for monthly, etc.).
    date_col : str, default='Order Date'
        Name of date column to aggregate on.

    Returns
    -------
    pd.DataFrame
        Aggregated time-series DataFrame indexed by date.
    """
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    df_copy = df_copy.set_index(date_col)

    agg_rules = {
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Order ID": "nunique"
    }
    
    # Filter rules to present columns
    rules = {k: v for k, v in agg_rules.items() if k in df_copy.columns}
    resampled = df_copy.resample(freq).agg(rules).rename(columns={"Order ID": "Orders"})
    
    # Fill gaps for daily/weekly if necessary
    resampled["Sales"] = resampled["Sales"].fillna(0.0)
    resampled["Profit"] = resampled["Profit"].fillna(0.0)
    resampled["Quantity"] = resampled["Quantity"].fillna(0)
    if "Orders" in resampled.columns:
        resampled["Orders"] = resampled["Orders"].fillna(0)

    resampled["Profit_Margin_Pct"] = np.where(
        resampled["Sales"] > 0,
        (resampled["Profit"] / resampled["Sales"]) * 100,
        0.0
    ).round(2)

    return resampled


def calculate_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics grouped by product Category.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame by Category.
    """
    cat = df.groupby("Category").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Transactions=("Sales", "count")
    )
    total_sales = df["Sales"].sum()
    cat["Sales_Share_Pct"] = ((cat["Sales"] / total_sales) * 100).round(2)
    cat["Profit_Margin_Pct"] = ((cat["Profit"] / cat["Sales"]) * 100).round(2)
    return cat.sort_values(by="Sales", ascending=False)


def calculate_subcategory_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics grouped by Sub-Category with Parent Category.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame by Sub-Category.
    """
    sub = df.groupby(["Category", "Sub-Category"]).agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Avg_Discount=("Discount", "mean"),
        Transactions=("Sales", "count")
    ).reset_index()

    sub["Profit_Margin_Pct"] = ((sub["Profit"] / sub["Sales"]) * 100).round(2)
    sub["Avg_Discount_Pct"] = (sub["Avg_Discount"] * 100).round(2)
    return sub.sort_values(by="Profit", ascending=False)


def calculate_regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics grouped by geographic Region.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame by Region.
    """
    reg = df.groupby("Region").agg(
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Orders=("Order ID", "nunique"),
        Transactions=("Sales", "count")
    )
    reg["Profit_Margin_Pct"] = ((reg["Profit"] / reg["Sales"]) * 100).round(2)
    return reg.sort_values(by="Sales", ascending=False)


def calculate_discount_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sales and profitability across discrete discount tiers.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame by discount tier.
    """
    df_copy = df.copy()
    bins = [-0.001, 0.0, 0.20, 0.40, 0.60, 0.80]
    labels = ["0% (No Discount)", "0.01% - 20%", "20.01% - 40%", "40.01% - 60%", "60.01% - 80%"]
    df_copy["Discount_Tier"] = pd.cut(df_copy["Discount"], bins=bins, labels=labels)

    tier_agg = df_copy.groupby("Discount_Tier", observed=False).agg(
        Transactions=("Sales", "count"),
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum")
    )
    tier_agg["Profit_Margin_Pct"] = np.where(
        tier_agg["Sales"] > 0,
        (tier_agg["Profit"] / tier_agg["Sales"]) * 100,
        0.0
    ).round(2)
    return tier_agg
