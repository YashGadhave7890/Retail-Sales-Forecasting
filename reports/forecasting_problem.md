# Forecasting Problem Definition

**Project:** Retail Sales Forecasting & Analytics  
**Phase:** Phase 7 – Forecasting Problem Definition & Feature Engineering  
**Status:** Completed & Validated  

---

## Business Objective
In modern retail operations, accurate demand and sales forecasts are vital for managing inventory replenishment, labor scheduling, procurement contracts, and working capital. The business objective is to forecast aggregate retail sales into the future with high accuracy, enabling retail managers to:
1. Prevent stock-outs during high-demand promotional surges (e.g. Q4 holiday peaks).
2. Minimize working capital lockup and holding costs during low-demand periods (e.g. January/February).
3. Align corporate staffing, warehouse logistics, and fulfillment capacity with expected demand fluctuations.

---

## Forecasting Target
- **Selected Target Variable:** Total Gross Sales Revenue (`Sales`), denominated in USD ($).
- **Target Nature:** Quantitative, continuous, strictly positive, temporally ordered, and historically observed across all four calendar years (2014–2017).
- **Transformation Note:** For machine learning model training, both the raw dollar target and its variance-stabilized log-transformed version $\ln(1 + \text{Sales})$ are supported.

---

## Time Granularity
- **Selected Granularity:** **Weekly Aggregation (`W-SUN`)** — weeks ending on Sunday.
- **Span:** Exactly 209 consecutive, contiguous calendar weeks from `2014-01-05` to `2017-12-31`.
- **Gaps / Missing Periods:** **0 missing weeks.** Every single week in the 4-year calendar contains confirmed retail transaction activity.

---

## Forecast Horizon
- **Operational Horizon ($H$):** **1 to 12 weeks ahead** (1 full fiscal quarter).
- **Short-Term Tactical Planning (1 to 4 weeks):** Informs weekly warehouse fulfillment, logistics dispatch, and regional store staffing.
- **Medium-Term Strategic Planning (5 to 12 weeks):** Informs quarterly inventory purchasing, supplier lead-time ordering, and promotional campaign budgeting.

---

## Unit of Prediction
- **Primary Prediction Unit:** Aggregate Company-Wide Retail Sales ($) per week.
- **Secondary / Hierarchical Unit (Extensible):** Category-level weekly retail sales (`Technology`, `Furniture`, `Office Supplies`).

---

## Why This Formulation Was Selected

1. **Alignment with Retail Decision Cycles:** 
   Retail supply chains, supplier orders, and store staff schedules are established on weekly cycles. Daily demand fluctuations are too noisy for medium-term purchasing contracts, while monthly planning is too coarse for operational logistics.
2. **Elimination of Intermittent Zero-Demand Gaps:**
   At the daily level, the dataset contains 221 zero-sales calendar days (15.16% zero-demand periods), primarily driven by weekends, national holidays, and carrier downtime. Daily modeling requires complex zero-inflated or intermittent-demand algorithms (e.g., Croston's method). Weekly aggregation resolves this challenge organically, yielding a 100% complete, non-zero time series (min weekly sales: $227.24, max: $38,176.81).
3. **Preservation of Robust Macro Seasonality:**
   Weekly aggregation dampens intraday and day-of-week noise while preserving the 51.6% Q4 holiday surge and March spring replenishment cycles identified in Phase 6 EDA.
4. **Statistical Power for ML Models:**
   With 209 contiguous weekly observations, the dataset provides sufficient sample size to perform robust rolling-origin cross-validation:
   - 2014 to 2016: 156 weeks (3 full years, 75% of data) for training and expanding-window validation.
   - 2017: 52 weeks (1 full year, 25% of data) for out-of-time walk-forward test evaluation.

---

## Alternative Formulations Considered

| Formulation | Granularity | Total Observations | Zero-Demand Gaps | Assessment & Reason for Rejection |
| :--- | :--- | :--- | :--- | :--- |
| **Daily Sales** | Daily (`D`) | 1,458 days | 221 days (15.16%) | **Rejected:** High noise-to-signal ratio; day-of-week volatility (e.g. Wednesday low vs. Monday/Friday high) obscures quarterly trends; requires zero-inflation imputation. |
| **Monthly Sales** | Monthly (`ME`) | 48 months | 0 months (0.0%) | **Rejected:** Only 48 observations is insufficient for training gradient boosting algorithms (XGBoost, LightGBM, Random Forest) without high risk of overfitting. |
| **Store / Item Level** | Daily Item ($N \times T$) | >1.8M potential cells | >95% zero-demand cells | **Rejected:** Extreme sparsity at individual SKU level (1,862 products across 5,009 orders) causes severe intermittent zero-demand across time. |
| **Weekly Company Sales (Selected)** | Weekly (`W-SUN`) | **209 weeks** | **0 weeks (0.0%)** | **Selected:** Optimal balance of continuous signal, zero-free series, retail operational alignment, and adequate ML training depth. |

---

## Limitations
1. **Macro vs. SKU Level Trade-Off:** Aggregating to company-wide weekly sales provides robust macro-level forecasts, but does not forecast shelf-level restocking requirements for individual SKUs.
2. **Sample Size Ceiling:** 209 observations is standard for weekly retail forecasting over 4 years, but necessitates parsimonious feature engineering to avoid the curse of dimensionality.
3. **Fixed Horizon Walk-Forward:** Models trained on weekly series assume a rolling walk-forward evaluation protocol rather than one-shot multi-year extrapolations.
