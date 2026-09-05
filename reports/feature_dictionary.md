# Feature Dictionary

**Project:** Retail Sales Forecasting & Analytics  
**Target Variable:** `Sales` (Weekly aggregate sales revenue in USD)  
**Dataset:** `data/processed/forecasting_features.csv`  

---

## Complete Feature Inventory and Leakage Risk Audit

| # | Feature Name | Data Type | Description | Available at Prediction Time? | Leakage Risk | Engineering Decision |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Date` | Date / String | Weekly period end date (Sunday) | Yes (Index) | None | Preserved as temporal index |
| 2 | `Sales` | Float | Total weekly sales revenue ($) | **TARGET** | Target Column | Preserved as primary prediction target |
| 3 | `log_sales` | Float | $\ln(1 + \text{Sales})$ | **TARGET** | Target Column | Preserved as variance-stabilized target |
| 4 | `year` | Integer | Calendar year (2015–2017) | Yes (Deterministic) | None | Included (Captures macro linear growth trend) |
| 5 | `quarter` | Integer | Fiscal/calendar quarter (1–4) | Yes (Deterministic) | None | Included (Captures quarterly fiscal cycles) |
| 6 | `month` | Integer | Calendar month (1–12) | Yes (Deterministic) | None | Included (Captures monthly seasonal wave) |
| 7 | `week_of_year` | Integer | ISO week number (1–53) | Yes (Deterministic) | None | Included (Captures weekly cadence) |
| 8 | `is_q4` | Binary (0/1) | Flag indicating peak holiday season (Sept–Dec) | Yes (Deterministic) | None | Included (Encodes verified 51.6% holiday surge) |
| 9 | `sin_week` | Float | $\sin(2\pi \cdot \text{week} / 52.18)$ | Yes (Deterministic) | None | Included (Smooth circular week periodicity) |
| 10 | `cos_week` | Float | $\cos(2\pi \cdot \text{week} / 52.18)$ | Yes (Deterministic) | None | Included (Smooth circular week periodicity) |
| 11 | `sin_month` | Float | $\sin(2\pi \cdot \text{month} / 12)$ | Yes (Deterministic) | None | Included (Smooth circular month periodicity) |
| 12 | `cos_month` | Float | $\cos(2\pi \cdot \text{month} / 12)$ | Yes (Deterministic) | None | Included (Smooth circular month periodicity) |
| 13 | `sales_lag_1` | Float | Sales at week $t-1$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures immediate 1-week momentum, $r=0.406$) |
| 14 | `sales_lag_2` | Float | Sales at week $t-2$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures 2-week persistence, $r=0.390$) |
| 15 | `sales_lag_3` | Float | Sales at week $t-3$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures 3-week persistence) |
| 16 | `sales_lag_4` | Float | Sales at week $t-4$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures 1-month prior baseline, $r=0.213$) |
| 17 | `sales_lag_8` | Float | Sales at week $t-8$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures 2-month prior baseline) |
| 18 | `sales_lag_12` | Float | Sales at week $t-12$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures 1-quarter prior baseline) |
| 19 | `sales_lag_52` | Float | Sales at week $t-52$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures same-week annual seasonality, $r=0.445$) |
| 20 | `sales_rolling_mean_4` | Float | Mean sales from $t-4$ to $t-1$ | Yes (Known at $t$) | Low (Shifted before rolling) | Included (Captures recent 1-month average run rate) |
| 21 | `sales_rolling_std_4` | Float | Std dev of sales from $t-4$ to $t-1$ | Yes (Known at $t$) | Low (Shifted before rolling) | Included (Captures recent 1-month demand volatility) |
| 22 | `sales_rolling_min_4` | Float | Minimum sales from $t-4$ to $t-1$ | Yes (Known at $t$) | Low (Shifted before rolling) | Included (Captures local lower support bound) |
| 23 | `sales_rolling_max_4` | Float | Maximum sales from $t-4$ to $t-1$ | Yes (Known at $t$) | Low (Shifted before rolling) | Included (Captures local upper ceiling) |
| 24 | `sales_rolling_mean_12`| Float | Mean sales from $t-12$ to $t-1$ | Yes (Known at $t$) | Low (Shifted before rolling) | Included (Captures medium-term quarterly trend) |
| 25 | `sales_rolling_std_12` | Float | Std dev of sales from $t-12$ to $t-1$ | Yes (Known at $t$) | Low (Shifted before rolling) | Included (Captures medium-term volatility) |
| 26 | `orders_lag_1` | Integer | Total purchase orders at week $t-1$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures immediate preceding transaction frequency) |
| 27 | `quantity_lag_1` | Integer | Total units sold at week $t-1$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures immediate preceding physical volume) |
| 28 | `profit_lag_1` | Float | Net profit generated at week $t-1$ | Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures preceding operational financial margin) |
| 29 | `avg_discount_lag_1` | Float | Average discount applied at week $t-1$| Yes (Known at $t$) | Low (Strictly shifted) | Included (Captures preceding promotional intensity) |

---

## Explicitly Excluded Contemporaneous Variables

| Variable | Reason for Exclusion | Target Leakage Risk if Included |
| :--- | :--- | :--- |
| `profit(t)` | Contemporaneous net profit is unknown until all transactions in week $t$ settle. | **FATAL LEAKAGE** |
| `quantity(t)` | Contemporaneous item quantity is co-determined with sales revenue. | **FATAL LEAKAGE** |
| `orders(t)` | Total order count during week $t$ is contemporaneous with week $t$ sales. | **FATAL LEAKAGE** |
| `discount(t)` | Contemporaneous promotional discounting rate is co-determined with week $t$ transactions. | **FATAL LEAKAGE** |
| `Ship Date` | Forward fulfillment date unknown prior to order placement. | **FATAL LEAKAGE** |
