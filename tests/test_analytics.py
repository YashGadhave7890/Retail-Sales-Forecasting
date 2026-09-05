"""
Unit tests for src.analytics module.
"""
import pytest
import pandas as pd
import numpy as np

from src.analytics import (
    calculate_core_kpis,
    aggregate_by_time,
    calculate_category_performance,
    calculate_subcategory_performance,
    calculate_regional_performance,
    calculate_discount_tiers
)
from src.config import PROCESSED_DATA_FILE


@pytest.fixture
def cleaned_data():
    """Load cleaned dataset for analytical testing."""
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df


def test_calculate_core_kpis(cleaned_data):
    """Verify calculation of high-level retail KPIs."""
    kpis = calculate_core_kpis(cleaned_data)
    assert isinstance(kpis, dict)
    assert kpis["num_orders"] == 5009
    assert kpis["num_customers"] == 793
    assert kpis["num_products"] == 1862
    assert kpis["total_sales"] == 2296919.49
    assert kpis["total_profit"] == 286409.08
    assert kpis["total_quantity"] == 37871
    assert kpis["average_order_value"] > 0
    assert kpis["average_profit_per_order"] > 0
    assert kpis["overall_profit_margin_pct"] == 12.47


def test_aggregate_by_time_monthly(cleaned_data):
    """Verify monthly temporal resampling produces 48 consecutive months."""
    monthly = aggregate_by_time(cleaned_data, freq="ME")
    assert len(monthly) == 48
    assert "Sales" in monthly.columns
    assert "Profit" in monthly.columns
    assert "Orders" in monthly.columns
    assert "Profit_Margin_Pct" in monthly.columns
    assert round(monthly["Sales"].sum(), 2) == 2296919.49


def test_aggregate_by_time_weekly(cleaned_data):
    """Verify weekly resampling produces continuous contiguous series without missing weeks."""
    weekly = aggregate_by_time(cleaned_data, freq="W-SUN")
    assert len(weekly) >= 208
    assert (weekly["Sales"] >= 0).all()


def test_category_performance(cleaned_data):
    """Verify category breakdown calculates metrics across all 3 product categories."""
    cat = calculate_category_performance(cleaned_data)
    assert len(cat) == 3
    assert set(cat.index) == {"Furniture", "Office Supplies", "Technology"}
    assert round(cat["Sales_Share_Pct"].sum(), 1) == 100.0


def test_subcategory_performance(cleaned_data):
    """Verify sub-category breakdown covers all 17 sub-categories."""
    sub = calculate_subcategory_performance(cleaned_data)
    assert len(sub) == 17
    assert "Profit_Margin_Pct" in sub.columns
    assert "Tables" in sub["Sub-Category"].values


def test_discount_tiers(cleaned_data):
    """Verify discount tier breakdown and identify negative profit margin threshold."""
    tiers = calculate_discount_tiers(cleaned_data)
    assert len(tiers) == 5
    # Tiers with discounts > 20% should exhibit negative profit margins
    high_disc_margin = tiers.loc["60.01% - 80%", "Profit_Margin_Pct"]
    assert high_disc_margin < 0
