# Retail Sales Forecasting Limitations & Methodological Constraints

## 1. Executive Summary

While the developed machine-learning forecasting pipeline (specifically **Ridge Regression** and regularized gradient-boosted ensembles) demonstrates verified predictive skill over naive and seasonal-naive baselines, this document outlines critical methodological, operational, and data-driven constraints.

> [!WARNING]
> **Not Production-Grade Autonomous Forecasting:**
> Outperforming persistence baselines on historical backtests confirms statistical learning, but does **not** make the model suitable for fully automated, autonomous replenishment without human oversight and inventory safety buffers.

---

## 2. Primary Data Constraints

### 2.1 Limited Historical Horizon
- **Duration:** The dataset spans exactly 4 calendar years (2014–2017).
- **Usable History:** Requiring a 52-week lag (`sales_lag_52`) to capture annual seasonal rhythms consumes the entire first year (2014), leaving only **157 usable weekly observations** (2015–2017).
- **Macroeconomic Shifts:** 3 years of usable history is insufficient to capture multi-year macroeconomic cycles, recessions, or long-term structural demand shifts.

### 2.2 Weekly Aggregation Trade-offs
- Aggregating daily transaction records into weekly totals (`W-SUN`) solves sparsity and weekend seasonality.
- **Limitation:** Intra-week volatility, daily stockout risks, weekend shopping surges, and day-of-week staffing needs cannot be inferred from weekly totals.

### 2.3 Single-Year Holdout Test Evaluation (2017)
- The holdout period consists of 53 weeks in calendar year 2017.
- While strictly held out from model development, performance on a single chronological year can be influenced by unique annual events (e.g., specific promotional timing, unusual weather, or supply chain bottlenecks in 2017).

---

## 3. Methodological & Architectural Constraints

### 3.1 Supervised One-Step-Ahead Formulation ($h=1$)
- The current supervised regression framework predicts $y_{t+1}$ given historical predictors up to $t$.
- **1–12 Week Planning Gap:** Retail inventory managers typically need multi-week planning horizons (e.g., 4 to 12 weeks forward).
- **Direct vs Recursive:** To produce multi-step forecasts ($h > 1$), this pipeline would either require recursive self-feeding of predicted lags (compounding forecasting errors) or multiple direct horizon models ($h=1, 2, \dots, 12$). Pretending the current model is a 12-week forecast without dynamic autoregressive rollouts is statistically invalid.

### 3.2 Absence of Exogenous Operational Covariates
The dataset consists solely of historical transactions without external explanatory variables:
1. **Promotional Calendars & Marketing Campaigns:** The model cannot anticipate upcoming advertising campaigns, discount coupons, or corporate catalog mailings; it only observes discounts after orders occur (`avg_discount_lag_1`).
2. **Specific Holiday Indicators:** While cyclical sine/cosine terms and `is_q4` capture seasonal periods, exact floating holidays (e.g., Labor Day, Thanksgiving, Cyber Monday, Easter) shift by several days each year, causing phase-shift errors.
3. **Inventory & Stockout Data:** Unfulfilled customer demand due to stockouts is recorded as $0$ sales, distorting true underlying demand.
4. **Competitor Pricing & Macroeconomic Conditions:** Inflation rates, consumer sentiment, and competitive discounting are unavailable.
5. **Weather & Geopolitical Factors:** Extreme winter storms or logistics disruptions cannot be modeled.

---

## 4. Empirical Error Magnitude & Operational Implications

### 4.1 Residual Error Variance
- **Holdout Test Performance:** Ridge Regression achieved a Test MAE of **$5,377.97** and a Test WAPE of **38.58%**.
- **Operational Reality:** A ~38% average percentage error means that weekly sales forecasts have an expected deviation of approximately ±$5,300 on an average weekly baseline of $13,939.
- For high-volume inventory procurement, a 38% error could lead to substantial overstocking or stockouts if treated as deterministic truth.

### 4.2 Asymmetric Peak vs Trough Under/Overprediction
- **Peak Underprediction:** During the top quartile of sales weeks (Q4 holiday rush, sales > $19,484), the model systematically underpredicts by an average of **+$5,810 per week**.
- **Post-Peak Overprediction:** Immediately following major demand spikes (e.g., early October and late December), the model overpredicts by an average of **-$7,079 per week** due to autoregressive lag inertia.

---

## 5. Summary Guidance for Deployment & Decision Support

| Decision Area | Guidance |
| :--- | :--- |
| **Tactical Horizon** | Use model strictly for 1–2 week near-term directional demand guidance. |
| **Safety Buffers** | One possible approach would be to use forecast uncertainty/error distributions to determine an appropriate safety margin rather than fixed percentage buffers, accounting for empirical peak underprediction (~$5,810 on weeks >$19,484). |
| **Post-Holiday Adjustment** | Manually discount automated forecasts following quarter-end corporate purchasing spikes. |
| **Model Retraining** | Retrain expanding models weekly as fresh Sunday close data becomes available. |
| **Human-in-the-Loop** | Maintain mandatory merchandising review before issuing purchase orders exceeding $25,000. |
