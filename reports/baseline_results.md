# Baseline Forecasting Model Results

## 1. Baseline Definitions

To benchmark genuine predictive skill before deploying machine-learning algorithms, two standard time-series baselines were evaluated on strictly chronological out-of-sample data:

1. **Naive Baseline (Lag-1 Persistence):**
   - **Formula:** $\hat{y}_t = y_{t-1}$ (`sales_lag_1`)
   - Assumes next week's revenue equals this week's revenue.
2. **Seasonal Naive Baseline (Lag-52 Persistence):**
   - **Formula:** $\hat{y}_t = y_{t-52}$ (`sales_lag_52`)
   - Assumes next week's revenue equals the revenue observed during the exact same calendar week of the previous year.
   - Evaluated strictly where 52 weeks of prior history exist (2015–2017).

---

## 2. Chronological Validation & Test Performance

All metrics are calculated empirically from real observations without synthetic adjustment.

| Metric | Naive Baseline (CV) | Naive Baseline (Test 2017) | Seasonal Naive (CV) | Seasonal Naive (Test 2017) |
| :--- | :--- | :--- | :--- | :--- |
| **MAE ($)** | $5,366.34 | **$7,187.28** | $5,342.45 | **$7,075.47** |
| **RMSE ($)** | $6,751.16 | **$9,134.80** | $6,864.83 | **$9,172.35** |
| **MAPE (%)** | 64.03% | **63.05%** | 55.80% | **61.50%** |
| **WAPE (%)** | 49.61% | **51.56%** | 49.14% | **50.76%** |

---

## 3. Key Observations

1. **High Baseline Error:** Both simple persistence baselines produce approximately **51% WAPE** on the 2017 holdout test set (MAE exceeds **$7,000** per week).
2. **Seasonal vs Lag-1:** Seasonal Naive slightly edges out Naive on Test MAE ($7,075.47 vs $7,187.28), demonstrating that annual seasonality provides meaningful signal over pure weekly inertia.
3. **Target Benchmark for Machine Learning:** To be considered genuinely useful, supervised machine learning models must substantially outperform these baseline thresholds:
   - **Target MAE:** < $7,075.47
   - **Target RMSE:** < $9,134.80
   - **Target WAPE:** < 50.76%

---

## 4. Visualizations

- Baseline comparison plot saved to: `assets/models/baseline_predictions.png`
- Individual plots: `assets/models/actual_vs_naive.png` and `assets/models/actual_vs_seasonal_naive.png`
