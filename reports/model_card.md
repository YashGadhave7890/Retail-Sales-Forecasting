# Model Card: Retail Sales Weekly Forecasting System

## 1. Model Overview & Purpose

- **Model Name:** Regularized Ridge Regression Forecaster
- **Model Type:** Scikit-learn Pipeline (`StandardScaler` + `Ridge(alpha=10.0, random_state=42)`)
- **Version:** 1.0.0 (Phase 10 Release)
- **Artifact Path:** `models/final_model.joblib`
- **Metadata Path:** `models/model_metadata.json`
- **Primary Objective:** Generate robust, statistically calibrated point forecasts of total retail sales revenue for the next weekly period ($h=1$ week) to support retail merchandising, inventory replenishment, and staffing planning.

---

## 2. Intended Users & Stakeholders

- **Primary Users:** Retail Demand Planners, Supply Chain Analysts, Inventory Procurement Managers.
- **Secondary Users:** Financial Planning & Analysis (FP&A) Teams, Merchandising Executives.
- **Deployment Context:** Executive decision-support tool integrated into weekly operational planning cadences; not intended for unmonitored closed-loop automated ordering.

---

## 3. Formulation & Target Specification

- **Target Variable ($y$):** Weekly aggregated Sales revenue in USD (`Sales`).
- **Temporal Granularity:** Weekly Sunday week-ending frequency (`W-SUN`).
- **Forecasting Formulation:** Supervised one-step-ahead ($h=1$) autoregressive regression.
- **Unit of Measurement:** Continuous currency (USD, $).
- **Target Transformations:** Features use untransformed and log-transformed revenue where appropriate, with model predicting raw USD sales directly.

---

## 4. Training & Evaluation Data

- **Data Source:** Publicly available Sample Superstore transaction dataset (2014–2017).
- **Historical Horizon:**
  - Raw: 9,993 transactions spanning 2014-01-03 to 2017-12-30.
  - Aggregated: 209 weekly periods.
  - Usable History: **157 weekly periods** (2015-01-04 to 2017-12-31) after requiring a 52-week lag history.
- **Evaluation Model Separation:**
  - *Evaluation Artifact (`models/ridge_regression.joblib`):* Trained strictly on 2015–2016 (104 weeks, 2015-01-04 to 2016-12-25) to provide unbiased benchmarking against the held-out 2017 test set.
  - *Production Artifact (`models/final_model.joblib`):* Trained on the entire 157-week dataset (2015–2017) to maximize coefficient precision for forward deployment.

---

## 5. Input Features & Leakage Architecture

The model utilizes **26 leakage-free engineered predictors**:

1. **Calendar & Seasonality (9 features):**
   `year`, `quarter`, `month`, `week_of_year`, `is_q4`, `sin_week`, `cos_week`, `sin_month`, `cos_month`.
2. **Autoregressive Lags (7 features):**
   `sales_lag_1`, `sales_lag_2`, `sales_lag_3`, `sales_lag_4`, `sales_lag_8`, `sales_lag_12`, `sales_lag_52`.
3. **Rolling Window Statistics (6 features):**
   `sales_rolling_mean_4`, `sales_rolling_std_4`, `sales_rolling_min_4`, `sales_rolling_max_4`, `sales_rolling_mean_12`, `sales_rolling_std_12` (all strictly shifted by 1 period before calculation).
4. **Lagged Business Drivers (4 features):**
   `orders_lag_1`, `quantity_lag_1`, `profit_lag_1`, `avg_discount_lag_1`.

### Leakage Controls:
- **No Target in Predictors:** Current week's Sales revenue ($y_t$) is strictly excluded from all rolling windows and features.
- **No Contemporaneous Information:** Unlagged operational metrics (same-week Profit, Quantity, Discount, Orders) are removed.
- **Strict Chronological Invariant:** $T_{\text{train, max}} < T_{\text{val/test, min}}$ enforced across all CV splits and holdout partitions.

---

## 6. Performance Metrics & Benchmark Comparisons

All evaluation metrics are computed on the untouched 2017 holdout test set (53 weekly periods):

| Model / Benchmark | Test MAE ($) | Test RMSE ($) | Test WAPE (%) | Test MAPE (%) | Beat Benchmark? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline ($t-1$)** | $7,187.28 | $9,134.80 | 51.56% | 63.05% | Reference |
| **Seasonal Naive Baseline ($t-52$)** | $7,075.47 | $9,172.35 | 50.76% | 61.50% | Reference |
| **Ridge Regression (Selected)** | **$5,377.97** | **$6,778.95** | **38.58%** | **56.84%** | **YES (+25.2%)** |
| Gradient Boosting | $5,493.08 | $7,125.67 | 39.41% | 47.31% | YES (+23.6%) |
| Random Forest | $5,600.20 | $7,289.91 | 40.18% | 46.25% | YES (+22.1%) |
| HistGradientBoosting | $5,926.80 | $7,601.07 | 42.52% | 49.76% | YES (+17.5%) |

### Cross-Validation Stability:
- 5-Fold Expanding Window Mean MAE: **$5,130.83** (Std: $1,684.12)
- 5-Fold Expanding Window Mean RMSE: **$6,554.35**
- 5-Fold Expanding Window Mean WAPE: **48.60%**

---

## 7. Model Strengths & Advantages

1. **Superior Regularization on Small Samples:** L2 penalty shrinks multi-lag collinearities smoothly, avoiding the overfit regimes of deep tree ensembles on 104–157 rows.
2. **Statistically Unbiased:** Rigorous one-sample t-test on residuals confirmed no global bias ($t = -1.4339, p = 0.1576$).
3. **Normally Distributed Residuals:** Shapiro-Wilk test confirmed residuals conform to a normal distribution ($W = 0.9751, p = 0.3302$).
4. **Zero Residual Autocorrelation:** Durbin-Watson statistic ($d = 2.0597$) confirms lag structure captured temporal serial correlation.
5. **Directional Interpretability:** Linear coefficients clearly indicate whether drivers positively or negatively impact sales.
6. **Minimal Computational Footprint:** Sub-millisecond inference and tiny artifact size (~2.5 KB).

---

## 8. Known Weaknesses & Limitations

1. **Systematic Holiday Peak Underprediction:** During extreme Q4 holiday spikes (>75th percentile sales), the model underpredicts actual demand by an average of **+$5,810 per week**.
2. **Post-Peak Slump Overprediction:** Following quarter-end corporate purchasing rushes, lag inertia causes overprediction by an average of **-$7,079 per week**.
3. **One-Step-Ahead Limitation:** The model is not designed for autonomous 12-week recursive forecasting without re-feeding predicted values.
4. **Point Forecasts:** Produces point estimates without native parametric confidence bands.
5. **Lack of Exogenous Covariates:** Lacks access to upcoming marketing campaign dates, coupon promotions, stockout records, weather events, and macroeconomic indicators.

---

## 9. Usage Guidelines & Governance

### Appropriate Use:
- Providing directional weekly baseline sales expectations 1–2 weeks in advance.
- Benchmarking regional demand patterns and supporting manual inventory reorder reviews.
- Serving as the statistical baseline for FP&A cash-flow modeling.

### Inappropriate Use:
- Autonomous, unmonitored execution of purchase orders or vendor contract commitments.
- Sub-weekly (daily/hourly) staffing or inventory allocation.
- Long-range multi-year strategic capital expenditure forecasting.
- Relying on point forecasts during Black Friday / Cyber Week without consulting merchandising promotional schedules and applying data-supported uncertainty buffers.
