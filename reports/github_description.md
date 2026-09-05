# GitHub Portfolio & Repository Presentation Guide

This guide provides verified, truthful repository metadata, presentation copy, resume bullet points, and social portfolio summaries for the **Retail Sales Forecasting & Analytics** project.

---

## 1. Short Repository Description (< 160 Characters)

> Retail sales forecasting & commercial analytics platform with leakage-safe ML pipelines, backtesting, residual diagnostics, and a Streamlit dashboard.

*(Character count: 147 characters)*

---

## 2. Suggested GitHub Topics

Add the following tags to the GitHub repository to optimize discoverability across Data Science, Machine Learning, and Time-Series topics:

- `retail-analytics`
- `sales-forecasting`
- `time-series`
- `machine-learning`
- `scikit-learn`
- `streamlit`
- `python`
- `data-science`
- `backtesting`
- `error-analysis`
- `docker`
- `feature-engineering`

---

## 3. Suggested Pinned Project Description (GitHub Profile)

**Retail Sales Forecasting & Analytics**
> Production-grade retail analytics and weekly sales forecasting engine built on commercial superstore transaction data. Implements strict leakage-free temporal feature engineering, benchmarks linear (Ridge) and non-linear (Random Forest, Gradient Boosting) algorithms against Naive baselines across expanding-window backtesting, performs residual diagnostics, and serves interactive predictions via an executive Streamlit dashboard.

---

## 4. Suggested Resume Bullet Points

### Option A: Technical / Machine Learning Focus
- **Developed an end-to-end retail sales forecasting system** using Ridge regression and gradient boosting ensembles, engineering 26 leakage-free calendar, autoregressive, and rolling features across 157 weekly observations.
- **Outperformed naive persistence baselines by 25.2%**, achieving a 38.58% WAPE ($5,377.97 holdout MAE) evaluated via 5-fold expanding-window chronological backtesting (`TimeSeriesSplit`).
- **Diagnosed commercial margin destruction patterns** across 9,993 transactions, identifying a -$156k margin collapse on orders discounted >20% and persistent holiday underprediction (~$5.8k/week).
- **Built and containerized an interactive Streamlit operations dashboard** with multi-horizon recursive forecasting, dynamic margin sensitivity toggles, and containerized Docker deployment specifications.

### Option B: Business Impact & Analytics Focus
- **Engineered an executive retail demand forecasting platform** that reduced forecast error (WAPE) from 51.56% to 38.58% over seasonal and naive baselines on a 2017 holdout test set.
- **Identified $156k in margin leakage** through econometric transaction analysis, uncovering an operational cliff where discounts above 20% caused gross margins to plummet from +29% to -84%.
- **Designed 5-fold rolling time-series backtests and residual diagnostic suites**, verifying normality (Shapiro-Wilk $p=0.33$), zero autocorrelation (Durbin-Watson $d=2.06$), and unbiased point predictions ($p=0.16$).
- **Deployed a self-service decision tool via Streamlit and Docker**, enabling inventory planners to model 1-to-12 week demand scenarios, evaluate stockout risks, and export planned orders.

---

## 5. Suggested LinkedIn / Portfolio Project Write-Up

### Headline / Post Title:
**Building an Interview-Defensible Retail Sales Forecasting & Analytics System**

### Post Body:
In retail supply chains, demand forecasting errors directly translate into lost revenue from stockouts or margin erosion from clearance markdowns. But real-world time-series data is full of hidden traps: unshifted rolling features, lookahead bias, and overfitting small temporal horizons.

I recently completed an end-to-end retail forecasting and analytics project using commercial transaction data (Sample Superstore). Here is what went into building it:

1. **Immutable Ingestion & Leakage-Safe Engineering:**
   - Ingested 9,994 raw transactions (verified via SHA-256 hash), preserving 1,870 negative-profit orders to accurately reflect loss-leader dynamics.
   - Built 26 leakage-safe weekly features: autoregressive lags (up to $t-52$), strictly shifted rolling statistics ($t-1$ exclusion), and circular calendar encodings.
   - Built an automated unit test suite that asserts mathematical shift invariants and chronological ordering.

2. **Model Benchmarking & Chronological Validation:**
   - Benchmarked regularized linear regression (Ridge) against non-linear tree ensembles (Random Forest, Gradient Boosting, HistGradientBoosting) and standard persistence baselines (Naive $t-1$ and Seasonal Naive $t-52$).
   - Used 5-fold expanding-window backtesting (`TimeSeriesSplit`) rather than random k-fold shuffling.
   - **Result:** Ridge Regression achieved the lowest holdout test MAE ($5,377.97) and WAPE (38.58%), reducing baseline error by 25.2% and demonstrating that regularized linear models outperform deeper trees when sample sizes are compact ($N=104$ training weeks).

3. **Residual Diagnostics & Business Insights:**
   - Evaluated 2017 holdout residuals: verified zero global bias ($p=0.158$), normal error distributions (Shapiro-Wilk $p=0.330$), and zero autocorrelation (Durbin-Watson $d=2.06$).
   - Highlighted operational realities: holiday Q4 peak demand is systematically underpredicted by ~$5,810/week due to Black Friday volatility, requiring dynamic safety stock margins rather than static assumptions.
   - Uncovered an econometric margin cliff: discounts $>20\%$ caused $156k in cumulative losses.

4. **Production Deployment & Engineering Rigor:**
   - Developed a 6-view interactive Streamlit application supporting multi-step recursive forecasting and dynamic scenario modeling.
   - Containerized via a multi-stage Dockerfile running on `python:3.11-slim` under a non-root security context with native health checking.
   - Backed by an exhaustive 80-test automated pytest suite.

Check out the full repository, documentation, and methodology here: [Link to GitHub]
