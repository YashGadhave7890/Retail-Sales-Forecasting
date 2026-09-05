# Retail Sales Forecasting & Analytics: Interview Preparation Cheat Sheet

This document contains concise, technically rigorous, and interview-defensible answers to the 17 core questions covering business context, machine learning methodology, validation architecture, and operational deployment.

---

### 1. What problem does the project solve?
It solves the retail supply chain dilemma between **stockouts** and **excess carrying costs**. Unpredicted demand surges during holiday peaks lead to lost revenue and stockouts, while overpredicting in post-holiday troughs causes over-ordering, working capital lockup, and margin-eroding clearance discounting. The project provides weekly demand point forecasts and commercial diagnostic playbooks to optimize inventory replenishment.

---

### 2. Why weekly forecasting?
Daily transaction volume in the dataset exhibited **17.6% zero-sales days** and high intra-week noise (e.g., weekend order clustering vs. weekday lulls). Aggregating to weekly Sunday week-ending periods (`W-SUN`) eliminates artificial zero-inflated targets, aligns naturally with standard retail supplier purchase order cycles (weekly replenishment), and provides a stable signal for medium-term inventory planning.

---

### 3. Why Sales as the target?
`Sales` (gross revenue in USD) represents the direct top-line monetary volume moving through the business and determines logistics capacity, warehouse throughput, and working capital requirements. While `Quantity` is useful for unit stocking, the extreme heterogeneity in catalog item sizes (e.g., a paperclip vs. a commercial copier) makes raw unit sums uninformative for aggregate logistics without product-level dimensioning.

---

### 4. Why Ridge was selected?
Ridge Regression ($L_2$ regularization with $\alpha=10.0$ and `StandardScaler`) achieved the best out-of-sample generalization across both 5-fold expanding-window backtesting (Mean MAE **$5,130.83**) and the out-of-sample 2017 holdout test set (MAE **$5,377.97**, RMSE **$6,778.95**, WAPE **38.58%**). With a compact sample size of 104 training weeks and 26 features, $L_2$ shrinkage effectively penalizes multicollinear autoregressive lags and rolling features without overfitting noisy outliers. Residual diagnostics confirmed zero statistically significant bias ($p=0.158$) and normal error distribution ($p=0.330$).

---

### 5. Why not Random Forest?
While Random Forest achieved competitive backtest performance (Mean MAE $5,196.86), it underperformed Ridge on the 2017 holdout test set (MAE **$5,600.20** vs. $5,377.97 for Ridge). Tree ensembles partition feature space with orthogonal axis cuts and cannot extrapolate trends or handle extreme holiday spikes outside the range of previous tree leaf splits without deep trees, which overfit volatile sub-samples on small datasets ($N=104$).

---

### 6. How did you prevent data leakage?
Three strict protocols were implemented:
1. **No Contemporaneous Features:** Same-week operational variables (`Profit`, `Quantity`, `Discount`, `Orders` at time $t$) were strictly excluded from predictor matrices.
2. **Explicit Shift Operations:** All autoregressive lags are shifted by at least 1 week ($t-1, t-2, \dots, t-52$). All rolling window statistics (e.g., 4-week and 12-week moving averages/standard deviations) are shifted by 1 period *before* computing the window, ensuring window $[t-k, t-1]$ never contains week $t$.
3. **Automated Leakage Audit:** Built `src/leakage_check.py` and `tests/test_feature_engineering.py` which programmatically verify exact 7-day timestamp increments, lag alignment, and target non-contamination.

---

### 7. Why chronological validation?
Standard randomized K-Fold cross-validation shuffles observations across time, causing future data points to predict past data points (lookahead bias) and dramatically understating true generalization error. We implemented a 5-fold expanding-window `TimeSeriesSplit` where for every fold, all training timestamps strictly precede all validation timestamps ($T_{\text{train, max}} < T_{\text{val, min}}$), perfectly mimicking how the model operates in live production.

---

### 8. What were the baseline models?
Two standard retail baselines were implemented:
1. **Naive Persistence Baseline ($t-1$):** Assumes next week's sales equal this week's sales ($\hat{y}_t = y_{t-1}$).
2. **Seasonal Naive Baseline ($t-52$):** Assumes sales equal the exact corresponding week from the prior year ($\hat{y}_t = y_{t-52}$).
On the 2017 holdout, Naive scored MAE $7,187.28 (WAPE 51.56%) and Seasonal Naive scored MAE $7,075.47 (WAPE 50.76%). The final Ridge model reduced MAE to $5,377.97 (WAPE 38.58%), demonstrating a **23.99% to 25.17% improvement over naive persistence**.

---

### 9. What was the final MAE?
On the unobserved 53-week 2017 holdout test set, the final Ridge model achieved a Mean Absolute Error of **$5,377.97** (with RMSE of **$6,778.95**).

---

### 10. What does 38.58% WAPE mean?
Weighted Absolute Percentage Error ($\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$) measures total absolute forecast error divided by total actual sales volume. Unlike traditional MAPE, which blows up to astronomical percentages when weekly sales are small, WAPE is volume-weighted and stable. A 38.58% WAPE means that across the entire 2017 test year ($733k in sales), total absolute prediction misses amounted to 38.58% of cumulative revenue—a 12.98 percentage point reduction compared to the 51.56% error of naive persistence.

---

### 11. What were the biggest model weaknesses?
Error analysis revealed an **asymmetric error profile across sales tiers**:
- **Peak Underprediction:** On high-demand weeks (>75th percentile, >$19,484), the model systematically underpredicted by an average of **+$5,809.98/week**. The maximum underprediction error was Black Friday week (2017-11-19), missing by **+$15,725.64**.
- **Trough Overprediction:** On low-demand weeks (<25th percentile, <$7,782), the model overpredicted by an average of **-$7,078.89/week** due to autoregressive lag inertia from prior active weeks.

---

### 12. Why is Q4 difficult?
Q4 accounts for **34.2% of annual sales**, concentrated into sharp, irregular event spikes (Black Friday, Cyber Monday, pre-Christmas shipping deadlines). Because the historical dataset contains only two prior training years (2015 and 2016), the model observed only two previous Q4 cycles. Event calendar shifts and promotion-driven demand spikes cannot be fully captured purely by smooth calendar harmonics without exogenous marketing/promotional data.

---

### 13. What would you improve with more data?
1. **Longer History (5–10 years):** More holiday cycles to distinguish macroeconomic trends from true seasonality.
2. **Promotional & Marketing Covariates:** Incorporating planned discount campaigns, circular ads, and email blitz dates.
3. **Inventory & Stockout Data:** Unobserved out-of-stock events censor true demand (unfulfilled orders appear as zero sales).
4. **Hierarchical Modeling:** Bottom-up forecasting at the Sub-Category/Store level reconciled up to total enterprise revenue via MinT (Minimum Trace) optimal reconciliation.

---

### 14. How is the model deployed?
The production pipeline is encapsulated into `src/forecasting.py`, which serializes the full dataset-trained Ridge model to `models/final_model.joblib` alongside schema and metric contracts in `models/model_metadata.json`. It is served interactively via a multi-page **Streamlit application** (`dashboard/app.py`), containerized via a production-grade multi-stage **Dockerfile** (running non-root on `python:3.11-slim` with internal health checks), and configured for **Streamlit Community Cloud**.

---

### 15. What is the role of Streamlit?
Streamlit serves as the **operational decision-support interface** for non-technical stakeholders (merchandisers, inventory managers, executives). It provides:
- Live KPI monitoring ($2.3M sales, margin cliffs).
- Interactive 1-to-12 week demand projections with dynamic recursive lag propagation.
- Side-by-side benchmark model evaluation.
- Scenario-based discount margin sensitivity analysis.
- One-click CSV export of operational demand plans.

---

### 16. What did Docker provide?
Docker provides **environment parity, reproducibility, and production isolation**:
- Standardizes Python 3.11 runtime and compiled C-extensions (NumPy, Scikit-learn).
- Eliminates "works on my machine" operating system differences.
- Implements container security best practices: running under an unprivileged user (`appuser`, UID 1000) and defining container health checks on port 8501.

---

### 17. What are the project's main limitations?
1. **Compact Sample Size:** 157 usable weekly observations after accounting for 52-week annual lags.
2. **Weekly Granularity:** Does not forecast daily store-level staffing or intra-day transaction timing.
3. **Recursive Lag Compounding:** Predictions for horizons $h > 1$ rely on recursive self-feeding of previous predictions, increasing variance over longer horizons.
4. **Absence of External Regressors:** No marketing spend, pricing elasticity curves, competitor actions, or inflation data.
5. **Guidance, Not Certainty:** Holdout WAPE is 38.58%; forecasts are strategic planning inputs, not deterministic guarantees.
