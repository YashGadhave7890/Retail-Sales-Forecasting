# Target Leakage Audit Report

**Dataset Audited:** `data/processed/forecasting_features.csv`  
**Total Observations:** 157 weekly periods  
**Total Features Evaluated:** 26 features  
**Audit Status:** PASS  

---

## Leakage Checks

The following automated integrity tests were performed:

1. **Strict Monotonic Ordering:** Verifies that row $i+1$ occurs chronologically after row $i$.
   - **Result:** PASS (Strictly increasing)
2. **Contiguous Weekly Cadence:** Verifies that interval between consecutive records is exactly 7 calendar days ($t_{i+1} - t_i = 7$).
   - **Result:** PASS (Uniform 7-day delta)
3. **No Duplicate Timestamps:** Verifies unique timestamps across all records.
   - **Result:** PASS (0 duplicate timestamps)
4. **Exact Lag Alignment:** Verifies that `sales_lag_1[i] == Sales[i-1]` for all rows.
   - **Result:** PASS (Exact mathematical identity)
5. **Rolling Window Pre-Shift:** Verifies that `sales_rolling_mean_4[i]` is computed over $[i-4, i-1]$ and strictly excludes $Sales[i]$.
   - **Result:** PASS (Confirmed shifted window)
6. **No Target Inclusion in Features:** Verifies that no rolling statistic contains the contemporaneous target value.
   - **Result:** PASS (Current target strictly excluded)
7. **Future Shift Contamination:** Verifies that no feature exhibits suspicious correlation ($|r| > 0.95$) with future target values $Sales[t+1]$.
   - **Result:** PASS (Zero anomalous future correlations)
8. **Contemporaneous Operational Leakage:** Verifies that raw unlagged `Profit`, `Quantity`, `Orders`, and `Discount` are excluded from the feature matrix.
   - **Result:** PASS (Zero forbidden contemporaneous variables)

---

## Features Investigated
All 27 engineered predictor variables were inspected:
- **Calendar & Periodic Features (9):** `year`, `quarter`, `month`, `week_of_year`, `is_q4`, `sin_week`, `cos_week`, `sin_month`, `cos_month`.
- **Autoregressive Lags (7):** `sales_lag_1`, `sales_lag_2`, `sales_lag_3`, `sales_lag_4`, `sales_lag_8`, `sales_lag_12`, `sales_lag_52`.
- **Historical Rolling Statistics (6):** `sales_rolling_mean_4`, `sales_rolling_std_4`, `sales_rolling_min_4`, `sales_rolling_max_4`, `sales_rolling_mean_12`, `sales_rolling_std_12`.
- **Operational Historical Lags (4):** `orders_lag_1`, `quantity_lag_1`, `profit_lag_1`, `avg_discount_lag_1`.

---

## Potential Leakage
- **Investigation of Operational Metrics:** In the raw weekly aggregation, contemporaneous variables `Profit`, `Quantity`, `Orders`, and `Avg_Discount` were calculated. Had they been included directly in row $t$, they would constitute severe target leakage because a retailer does not know total profit or customer volume for next week in advance.
- **Investigation of Rolling Statistics:** If pandas `.rolling(4).mean()` had been executed without `.shift(1)`, row $t$ would incorporate $Sales[t]$ into its own input feature, causing the model to learn a trivial identity rather than a forecast.

---

## Corrections
1. **Applied `.shift(1)` Pre-Rolling:** Enforced `df['Sales'].shift(1).rolling(W)` across all rolling features (`mean`, `std`, `min`, `max`), ensuring the current period is entirely absent from the window.
2. **Purged Contemporaneous Operational Metrics:** Stripped unlagged `Profit`, `Quantity`, `Orders`, and `Avg_Discount` from the feature matrix, preserving only their strictly shifted historical counterparts (`_lag_1`).
3. **Explicit Log Target Separation:** Stored `log_sales` purely as an alternate target column and isolated it from predictor inputs.

---

## Final Decision
**NO TARGET LEAKAGE DETECTED**

All automated leakage tests passed with zero violations. The dataset `data/processed/forecasting_features.csv` is certified leak-free and ready for model training.
