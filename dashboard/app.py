"""
Retail Sales Forecasting & Analytics — Interactive Streamlit Dashboard.

Provides executive analytics, slice-and-dice sales filtering, recursive multi-step
demand forecasting, historical model benchmarking, and strategic business playbooks.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from dashboard.data_utils import (
    load_transaction_data,
    load_forecasting_features_data,
    get_cached_model,
    load_evaluation_reports,
    compute_executive_kpis,
    recursive_multistep_forecast
)
from src.config import REPORTS_DIR, ASSETS_MODELS_DIR, MODELS_DIR

# ====================================================================
# PAGE CONFIGURATION & STYLING
# ====================================================================
st.set_page_config(
    page_title="Retail Sales Forecasting & Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional typography and card styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ====================================================================
# DATA INGESTION
# ====================================================================
with st.spinner("Loading analytical data and forecasting model..."):
    trans_df = load_transaction_data()
    features_df = load_forecasting_features_data()
    model = get_cached_model()
    eval_reports = load_evaluation_reports()


# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
with st.sidebar:
    st.markdown("### 🛒 Retail Intelligence")
    st.markdown("**Retail Sales Forecasting & Analytics**")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Sales Analytics",
            "Forecast",
            "Model Performance",
            "Business Insights",
            "About / Limitations"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚙️ System Status")
    st.markdown("""
    - **Dataset:** Sample Superstore
    - **Records:** 9,993 Cleaned Orders
    - **Weekly Series:** 157 Weeks (2015–2017)
    - **Model:** Ridge Regression (L2)
    - **Holdout MAE:** $5,377.97 (WAPE 38.6%)
    """)
    st.markdown("---")
    st.caption("Version 1.0 • Professional Portfolio Project")


# ====================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ====================================================================
if selected_page == "Executive Overview":
    st.markdown('<div class="main-header">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Commercial performance summary and verified high-level retail KPIs (2014–2017).</div>', unsafe_allow_html=True)
    
    # Compute verified KPIs
    kpis = compute_executive_kpis(trans_df)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Revenue</div><div class="kpi-value">${kpis["total_sales"]:,.0f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Profit</div><div class="kpi-value">${kpis["total_profit"]:,.0f}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Profit Margin</div><div class="kpi-value">{kpis["profit_margin"]:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Units Sold</div><div class="kpi-value">{kpis["total_units"]:,}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Orders Placed</div><div class="kpi-value">{kpis["total_orders"]:,}</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Order Value</div><div class="kpi-value">${kpis["aov"]:.2f}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chart Row 1: Annual Revenue & Profit Progression
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        st.markdown("#### Annual Revenue & Profit Growth")
        annual_df = trans_df.groupby("Year").agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        ).reset_index()
        annual_df["Year"] = annual_df["Year"].astype(str)
        
        # Display as a clean grouped bar chart
        st.bar_chart(annual_df.set_index("Year")[["Sales", "Profit"]], height=320)
        st.caption("Annual sales expanded from $484k in 2014 to $733k in 2017 (+51.4% overall growth).")
        
    with col_b:
        st.markdown("#### Category Contribution")
        cat_df = trans_df.groupby("Category").agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        ).reset_index()
        cat_df["Margin (%)"] = np.round((cat_df["Profit"] / cat_df["Sales"]) * 100, 1)
        
        st.dataframe(
            cat_df.style.format({
                "Sales": "${:,.2f}",
                "Profit": "${:,.2f}",
                "Margin (%)": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.info("💡 **Key Finding:** Furniture generates 32.3% of company revenue but yields only a 2.49% margin due to loss-leader tables and bookcases.")
        
    # Chart Row 2: Regional Performance & Monthly Seasonality
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("#### Geographic Regional Performance")
        reg_df = trans_df.groupby("Region").agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        ).reset_index()
        st.bar_chart(reg_df.set_index("Region")[["Sales", "Profit"]], height=300)
        st.caption("West and East regions drive 60.9% of company revenue and 69.8% of net profit.")
        
    with col_d:
        st.markdown("#### Monthly Demand Seasonality (All Years)")
        monthly_season = trans_df.groupby("Month").agg(Sales=("Sales", "sum")).reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_season["Month_Name"] = [month_names[m-1] for m in monthly_season["Month"]]
        st.line_chart(monthly_season.set_index("Month_Name")["Sales"], height=300)
        st.caption("Demand surges dramatically in September, November, and December, representing 41.6% of annual sales.")


# ====================================================================
# PAGE 2: SALES ANALYTICS
# ====================================================================
elif selected_page == "Sales Analytics":
    st.markdown('<div class="main-header">Interactive Sales Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Slice and dice transactions by time period, product taxonomy, customer segment, and geography.</div>', unsafe_allow_html=True)
    
    # Filter Bar
    with st.expander("🔍 Filter Controls & Segment Selection", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            all_years = ["All"] + sorted(trans_df["Year"].unique().tolist())
            sel_year = st.selectbox("Order Year", all_years)
            
        with f_col2:
            all_cats = ["All"] + sorted(trans_df["Category"].unique().tolist())
            sel_cat = st.selectbox("Category", all_cats)
            
        with f_col3:
            all_segs = ["All"] + sorted(trans_df["Segment"].unique().tolist())
            sel_seg = st.selectbox("Customer Segment", all_segs)
            
        with f_col4:
            all_regs = ["All"] + sorted(trans_df["Region"].unique().tolist())
            sel_reg = st.selectbox("Geographic Region", all_regs)
            
    # Apply Filtering
    filtered_df = trans_df.copy()
    if sel_year != "All":
        filtered_df = filtered_df[filtered_df["Year"] == sel_year]
    if sel_cat != "All":
        filtered_df = filtered_df[filtered_df["Category"] == sel_cat]
    if sel_seg != "All":
        filtered_df = filtered_df[filtered_df["Segment"] == sel_seg]
    if sel_reg != "All":
        filtered_df = filtered_df[filtered_df["Region"] == sel_reg]
        
    # Filtered KPIs
    f_kpis = compute_executive_kpis(filtered_df)
    k_c1, k_c2, k_c3, k_c4 = st.columns(4)
    with k_c1:
        st.metric("Filtered Sales", f"${f_kpis['total_sales']:,.2f}")
    with k_c2:
        st.metric("Filtered Profit", f"${f_kpis['total_profit']:,.2f}")
    with k_c3:
        st.metric("Profit Margin", f"{f_kpis['profit_margin']:.1f}%")
    with k_c4:
        st.metric("Transactions", f"{len(filtered_df):,}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### Monthly Sales Over Time")
        time_grp = filtered_df.groupby("YearMonth")["Sales"].sum().reset_index()
        st.line_chart(time_grp.set_index("YearMonth")["Sales"], height=320)
        
    with t_col2:
        st.markdown("#### Sub-Category Profitability Breakdown")
        sub_grp = filtered_df.groupby("Sub-Category")["Profit"].sum().sort_values().reset_index()
        
        # Color bar indicating loss vs profit
        fig, ax = plt.subplots(figsize=(8, 5.5))
        bar_colors = ["#d62728" if p < 0 else "#2ca02c" for p in sub_grp["Profit"]]
        ax.barh(sub_grp["Sub-Category"], sub_grp["Profit"], color=bar_colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("Cumulative Net Profit ($)")
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)
        plt.close(fig)
        
    st.markdown("#### Top States by Filtered Volume")
    state_df = filtered_df.groupby("State").agg(
        Orders=("Order ID", "nunique"),
        Sales=("Sales", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Sales", ascending=False).head(10).reset_index()
    
    st.dataframe(
        state_df.style.format({
            "Sales": "${:,.2f}",
            "Profit": "${:,.2f}",
            "Orders": "{:,}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ====================================================================
# PAGE 3: FORECAST
# ====================================================================
elif selected_page == "Forecast":
    st.markdown('<div class="main-header">Weekly Sales Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production forecasting engine powered by regularized Ridge Regression.</div>', unsafe_allow_html=True)
    
    tab_future, tab_holdout = st.tabs(["🔮 Future Horizon Projection (2018)", "🧪 Historical Holdout Evaluation (2017)"])
    
    with tab_future:
        st.markdown("#### Out-of-Sample Forward Planning Projection")
        st.write("Generate forward weekly point forecasts starting from the last available Sunday (`2017-12-31`).")
        
        f_col_left, f_col_right = st.columns([1, 3])
        
        with f_col_left:
            horizon_sel = st.slider("Forecast Horizon (Weeks Ahead)", min_value=1, max_value=12, value=8)
            st.info("""
            **Formulation Notice:**
            - **Week 1 ($h=1$):** Direct supervised forecast using verified historical lags.
            - **Weeks 2–12 ($h>1$):** Generated via sequential recursive lag propagation.
            - Point forecasts only; no synthetic confidence bands.
            """)
            
        with f_col_right:
            # Generate recursive forecast
            fc_df = recursive_multistep_forecast(features_df, model, horizon_weeks=horizon_sel)
            
            # Summary metrics
            total_proj = fc_df["predicted_sales"].sum()
            avg_proj = fc_df["predicted_sales"].mean()
            peak_fc = fc_df.loc[fc_df["predicted_sales"].idxmax()]
            
            m_c1, m_c2, m_c3 = st.columns(3)
            with m_c1:
                st.metric("Total Projected Sales", f"${total_proj:,.2f}")
            with m_c2:
                st.metric("Avg Weekly Projected Run-Rate", f"${avg_proj:,.2f}")
            with m_c3:
                st.metric("Peak Projected Week", f"${peak_fc['predicted_sales']:,.2f}", f"{peak_fc['forecast_date']}")
                
            # Line chart of forecast
            chart_df = fc_df.set_index("forecast_date")[["predicted_sales"]]
            st.line_chart(chart_df, height=320)
            
        st.markdown("#### Tabular Forecast Output")
        st.dataframe(
            fc_df.style.format({"predicted_sales": "${:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        # CSV Download Button
        csv_bytes = fc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_bytes,
            file_name=f"sales_forecast_{horizon_sel}weeks.csv",
            mime="text/csv"
        )
        
    with tab_holdout:
        st.markdown("#### 2017 Chronological Holdout Evaluation Benchmark")
        st.write("Performance of the model on the strictly held-out final year (53 weekly periods).")
        
        # Load 2017 predictions
        test_mask = features_df["Date"].dt.year == 2017
        test_df = features_df[test_mask].copy()
        
        feature_cols = [c for c in features_df.columns if c not in ["Date", "Sales", "log_sales"]]
        y_test = test_df["Sales"].values
        y_pred = model.predict(test_df[feature_cols])
        
        eval_df = pd.DataFrame({
            "Date": test_df["Date"].dt.strftime("%Y-%m-%d"),
            "Actual Sales": np.round(y_test, 2),
            "Predicted Sales": np.round(y_pred, 2),
            "Absolute Error": np.round(np.abs(y_test - y_pred), 2)
        })
        
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.metric("Holdout Test MAE", "$5,377.97", "-25.2% vs Naive")
        with h_col2:
            st.metric("Holdout Test RMSE", "$6,778.95", "-25.8% vs Naive")
        with h_col3:
            st.metric("Holdout Test WAPE", "38.58%", "-13.0% vs Naive")
            
        st.line_chart(eval_df.set_index("Date")[["Actual Sales", "Predicted Sales"]], height=340)
        st.caption("Observed 2017 Actuals vs Model Point Predictions across all 53 weeks.")


# ====================================================================
# PAGE 4: MODEL PERFORMANCE
# ====================================================================
elif selected_page == "Model Performance":
    st.markdown('<div class="main-header">Model Benchmarking & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Verified historical evaluation results across baselines and machine-learning regressors.</div>', unsafe_allow_html=True)
    
    st.markdown("#### Out-of-Sample Model Comparison (2017 Holdout Test Year)")
    if "comparison_df" in eval_reports:
        comp_df = eval_reports["comparison_df"]
        st.dataframe(
            comp_df.style.format({
                "CV MAE": "${:,.2f}",
                "CV RMSE": "${:,.2f}",
                "CV MAPE (%)": "{:.2f}%",
                "CV WAPE (%)": "{:.2f}%",
                "Test MAE": "${:,.2f}",
                "Test RMSE": "${:,.2f}",
                "Test MAPE (%)": "{:.2f}%",
                "Test WAPE (%)": "{:.2f}%",
                "Train Time (s)": "{:.4f}s"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Bar comparison chart
        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### Holdout Test MAE by Model ($)")
            mae_chart = comp_df.set_index("Model")[["Test MAE"]]
            st.bar_chart(mae_chart, height=300)
            
        with col_m2:
            st.markdown("##### Holdout Test WAPE by Model (%)")
            wape_chart = comp_df.set_index("Model")[["Test WAPE (%)"]]
            st.bar_chart(wape_chart, height=300)
            
    st.markdown("---")
    st.markdown("#### 5-Fold Expanding-Window Backtesting Results")
    if "backtest_df" in eval_reports:
        b_df = eval_reports["backtest_df"]
        b_summary = b_df.groupby("Model").agg(
            Mean_MAE=("MAE", "mean"),
            Median_MAE=("MAE", "median"),
            Std_MAE=("MAE", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_WAPE=("WAPE", "mean")
        ).reset_index().sort_values("Mean_MAE")
        
        st.dataframe(
            b_summary.style.format({
                "Mean_MAE": "${:,.2f}",
                "Median_MAE": "${:,.2f}",
                "Std_MAE": "${:,.2f}",
                "Mean_RMSE": "${:,.2f}",
                "Mean_WAPE": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Expanding-window backtest confirms Ridge Regression achieved the lowest cross-fold error ($5,130.83 Mean MAE).")


# ====================================================================
# PAGE 5: BUSINESS INSIGHTS
# ====================================================================
elif selected_page == "Business Insights":
    st.markdown('<div class="main-header">Strategic Business Insights & Decision Playbook</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical observations, behavioral interpretations, and risk-adjusted executive recommendations.</div>', unsafe_allow_html=True)
    
    with st.expander("1. 📈 Multi-Year Revenue Growth (+51.4% Compounding)", expanded=True):
        st.markdown("""
        - **OBSERVATION:** Total 4-year sales reached $2,297,200.86, growing from $484k in 2014 to $733k in 2017 (+51.4% overall expansion).
        - **INTERPRETATION:** Retail expansion accelerated strongly in 2016–2017 due to expanding B2B corporate and home-office customer accounts.
        - **RECOMMENDATION:** Maintain commercial momentum while closely tracking cohort retention rather than assuming perpetual 20% compounding without capital investment.
        """)
        
    with st.expander("2. 🎄 Intense Q4 Demand Seasonality (34.2% of Annual Volume)", expanded=True):
        st.markdown("""
        - **OBSERVATION:** Q4 alone generates 34.2% of total company revenue. September (12.1%), November (15.3%), and December (14.2%) represent 41.6% of annual volume, while Jan/Feb represent demand troughs (~4.3%).
        - **INTERPRETATION:** B2B fiscal year-end budget exhaustion coincides with B2C holiday shopping promotions, creating massive concentrated transaction spikes.
        - **RECOMMENDATION:** Align carrier freight contracts and temporary warehouse labor around September and November/December peaks; schedule systems maintenance during the January volume lull.
        """)
        
    with st.expander("3. 📉 The 20% Discount Cliff (1,870 Negative Margin Orders)", expanded=True):
        st.markdown("""
        - **OBSERVATION:** Undiscounted orders deliver +29.8% margin; 10–20% discounts deliver +15.6% margin; discounts >20% collapse into deep losses (-12.4% to -84.2%), causing -$156k in lost margin.
        - **INTERPRETATION:** Commercial sales representatives use deep discounts (>20%) as a blunt tool to meet gross revenue targets without unit margin governance.
        - **RECOMMENDATION:** Establish a hard governance rule requiring executive VP approval for commercial discounts exceeding 20%, and tie sales compensation to gross margin contribution.
        """)
        
    with st.expander("4. 🪑 Product Category Imbalances: Furniture Loss-Leaders", expanded=True):
        st.markdown("""
        - **OBSERVATION:** Technology generated $145.5k profit (17.4% margin) and Office Supplies delivered $122.5k profit (17.0% margin). Furniture generated $742k revenue but only $18.5k profit (2.49% margin), heavily weighed down by Tables (-$17.7k loss) and Bookcases (-$3.5k loss).
        - **INTERPRETATION:** Bulky freight surcharges and aggressive unbundled discounting completely destroy operating margins on oversized furniture.
        - **RECOMMENDATION:** Eliminate standalone discounting on Tables; bundle Furniture exclusively with high-margin Technology hardware (Copiers, Displays).
        """)
        
    with st.expander("5. 🎯 Forecasting Peak/Trough Error Patterns & Planning Guidance", expanded=True):
        st.markdown("""
        - **OBSERVATION:** The production model is globally unbiased ($p=0.158$), but systematically underpredicts holiday peak weeks (>75th percentile sales) by an average of +$5,810/week and overpredicts post-holiday slumps by -$7,079/week.
        - **INTERPRETATION:** Linear autoregressive models suffer inertia after extreme surges and cannot bend sharply enough to capture 3× non-linear promotional spikes.
        - **RECOMMENDATION (Planning Safety Margin):** One possible approach would be to use forecast uncertainty/error distributions to determine an appropriate safety margin. For example, during high-sales periods, planners should account for empirical peak underprediction (~22.6% of peak sales) rather than assuming a static flat percentage buffer.
        """)


# ====================================================================
# PAGE 6: ABOUT / LIMITATIONS
# ====================================================================
elif selected_page == "About / Limitations":
    st.markdown('<div class="main-header">About the Project & Methodological Constraints</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Technical specifications, data lineage, and transparent forecasting limitations.</div>', unsafe_allow_html=True)
    
    st.markdown("#### Project Architecture & Formulation")
    st.markdown("""
    - **Dataset:** Sample Superstore retail transactions (9,993 orders, 2014–2017).
    - **Target Variable:** Weekly aggregated Sales revenue in USD (`Sales`, frequency `W-SUN`).
    - **Forecasting Setup:** Supervised one-step-ahead ($h=1$) autoregressive regression using 26 leakage-free predictors (calendar cyclical encodings, autoregressive sales lags, rolling statistics, and lagged operational drivers).
    - **Production Model:** Scikit-learn Pipeline with `StandardScaler` and `Ridge(alpha=10.0, random_state=42)`.
    """)
    
    st.markdown("---")
    st.markdown("#### Documented Forecasting Limitations")
    st.markdown("""
    1. **Small Sample Horizon:** Only 4 calendar years exist in total; requiring a 52-week lag history leaves exactly **157 usable weekly observations** (2015–2017).
    2. **Weekly Aggregation Level:** Intra-week daily volatility, weekend shopping spikes, and daily stockout risks are smoothed out.
    3. **Supervised 1-Step-Ahead Formulation:** The native architecture predicts 1 week forward. Multi-step horizons (2–12 weeks) require recursive self-feeding of predicted lags, which carries error propagation.
    4. **Absence of Exogenous Covariates:** The dataset lacks promotional campaign schedules, coupon marketing calendars, stockout/inventory records, weather data, competitor pricing, and macroeconomic indices.
    5. **Forecast Error Magnitude:** Holdout WAPE is 38.58% (MAE: $5,377.97). While beating naive persistence by 25.2%, an average error of ±$5,300 means forecasts must be treated as directional guidance rather than deterministic truth.
    """)
    
    st.warning("""
    ⚠️ **Operational Governance Disclaimer:**
    This dashboard is an analytical and decision-support tool. Automated forecasts must **always be combined with current commercial and operational intelligence** (merchandising calendars, vendor lead times, warehouse capacity) prior to executing inventory procurement or financial commitments.
    """)
