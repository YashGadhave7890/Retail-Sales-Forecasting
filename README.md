# Retail Sales Forecasting & Analytics

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-80%20Passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Model-Ridge%20Regression-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

An end-to-end, interview-defensible Data Science and Machine Learning project that analyzes real commercial retail transactions, diagnoses margin destruction patterns, engineers leakage-free time-series features, benchmarks multiple forecasting algorithms against rigorous baselines, evaluates models using expanding-window backtesting, and delivers an interactive operational dashboard.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Key Objectives](#3-key-objectives)
4. [Dataset & Lineage](#4-dataset--lineage)
5. [Data Cleaning Pipeline](#5-data-cleaning-pipeline)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Forecasting Problem Formulation](#7-forecasting-problem-formulation)
8. [Leakage-Free Feature Engineering](#8-leakage-free-feature-engineering)
9. [Target Leakage Prevention & Auditing](#9-target-leakage-prevention--auditing)
10. [Baselines & Machine Learning Models](#10-baselines--machine-learning-models)
11. [Chronological Validation & Backtesting](#11-chronological-validation--backtesting)
12. [Final Model Selection](#12-final-model-selection)
13. [Actual Performance Metrics](#13-actual-performance-metrics)
14. [Error Analysis & Residual Diagnostics](#14-error-analysis--residual-diagnostics)
15. [Strategic Business Insights](#15-strategic-business-insights)
16. [Streamlit Interactive Dashboard](#16-streamlit-interactive-dashboard)
17. [Repository Structure](#17-repository-structure)
18. [Installation & Setup](#18-installation--setup)
19. [Running Locally](#19-running-locally)
20. [Docker Containerization](#20-docker-containerization)
21. [Cloud Deployment Guide](#21-cloud-deployment-guide)
22. [Automated Test Suite](#22-automated-test-suite)
23. [Methodological Limitations](#23-methodological-limitations)
24. [Future Enhancements](#24-future-enhancements)
25. [License](#25-license)

---

## 1. Project Overview

Forecasting customer demand in retail environments is notoriously prone to methodological traps: lookahead bias, unshifted target leakage, arbitrary train/test shuffling, and black-box overfitting. 

This project demonstrates a production-grade machine learning lifecycle on the **Sample Superstore** dataset:
- Rigorous data cleaning that preserves operational realities (such as negative-profit loss leaders).
- Econometric exploratory data analysis uncovering structural margin cliffs.
- Strict chronological time-series validation across 157 weekly periods (2015–2017).
- Benchmark comparisons proving that machine learning outperforms naive persistence.
- Transparent reporting of model limitations and error distributions without exaggerated claims.

---

## 2. Business Problem

Retail operations face dual supply chain risks:
1. **Stockouts & Lost Revenue:** Unanticipated demand surges during holiday peaks result in unfulfilled orders and degraded brand trust.
2. **Excess Carrying Costs & Margin Erosion:** Inaccurate post-peak forecasts cause inventory overstocking, tied-up working capital, and forced clearance discounting.

The goal is to provide reliable, statistically unbiased weekly sales point forecasts that empower merchandising and supply chain planners to schedule inventory reorders and warehouse staffing effectively.

---

## 3. Key Objectives

- **Data Integrity:** Establish an immutable raw ingestion layer and reproducible cleaning pipeline.
- **Leakage Prevention:** Guarantee that no future target information or contemporaneous operational metrics leak into predictor feature matrices.
- **Rigorous Benchmarking:** Evaluate ML models against standard Naive ($t-1$) and Seasonal Naive ($t-52$) baselines.
- **Robustness Auditing:** Validate stability using 5-fold expanding-window `TimeSeriesSplit` and in-depth residual hypothesis testing.
- **Decision Support:** Translate predictive performance into an interactive Streamlit application and strategic executive playbooks.

---

## 4. Dataset & Lineage

- **Dataset:** Sample Superstore retail transaction dataset.
- **Raw File:** `data/raw/Sample - Superstore.csv` (SHA-256: `c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce`).
- **Scope:** 9,994 raw rows spanning 2014-01-03 to 2017-12-30 (4 full calendar years).
- **Attributes:** 21 features covering order identifiers, customer segments, geography (City, State, Region), product taxonomy (Category, Sub-Category), and transaction metrics (Sales, Quantity, Discount, Profit).

---

## 5. Data Cleaning Pipeline

The cleaning pipeline (`src/data_cleaning.py`) produces `data/processed/superstore_cleaned.csv`:
- **Deduplication:** Exactly 1 genuine duplicate record (Row ID 3407, identical clone of Row ID 3406) was removed, yielding **9,993 valid transactions**.
- **Preservation of Reality:** Exactly **1,870 negative-profit transactions** were strictly preserved. Deleting unprofitable orders is common bad practice that blinds models to real commercial losses.
- **Data Normalization:** Column names and string values were trimmed; dates were validated to fall within `[2014-01-01, 2017-12-31]`; postal codes were formatted as 5-digit strings.

---

## 6. Exploratory Data Analysis (EDA)

Key commercial patterns uncovered (`reports/eda_report.md`):
1. **Revenue Growth:** Annual revenue grew from $484k (2014) to $733k (2017), an overall increase of **+51.4%**. Total cumulative sales reached **$2,296,919.49** with net profit of **$286,409.08** (12.47% margin).
2. **Q4 Demand Clustering:** Q4 alone generates **34.2%** of company revenue. September (12.1%), November (15.3%), and December (14.2%) represent **41.6%** of all sales.
3. **Category Imbalances:**
   - **Technology:** $836k sales, $145.5k profit (**17.4% margin**); star driver is *Copiers* ($55.6k profit, 36.4% margin).
   - **Office Supplies:** $719k sales, $122.5k profit (**17.0% margin**).
   - **Furniture:** $742k sales, but only $18.5k profit (**2.49% margin**), heavily depressed by loss-leaders: *Tables* (-$17.7k loss) and *Bookcases* (-$3.5k loss).
4. **The 20% Discount Margin Cliff:** Orders with discounts $\le 20\%$ yield positive margins (+15.6% to +29.8%). Discounts $> 20\%$ collapse into catastrophic losses (-12.4% to -84.2%), resulting in **-$156,131.29** in cumulative lost margin.

---

## 7. Forecasting Problem Formulation

- **Target Variable:** Total weekly Sales revenue in USD (`Sales`).
- **Frequency:** Weekly Sunday week-ending (`W-SUN`). Daily series exhibited 17.6% zero-sales days; weekly aggregation eliminates artificial zeroes and smooths day-of-week noise.
- **Historical Horizon:** 209 total weekly observations. Reserving 52 weeks for the annual seasonal lag leaves **157 usable weekly periods** (2015-01-04 to 2017-12-31).
- **Chronological Split:**
  - **Training / Model Development:** 104 weeks (2015-01-04 to 2016-12-25, 66.2%).
  - **Holdout Test Set:** 53 weeks (2017-01-01 to 2017-12-31, 33.8%).

---

## 8. Leakage-Free Feature Engineering

The feature engineering pipeline (`src/feature_engineering.py`) produces `data/processed/forecasting_features.csv` with **26 leakage-free predictors**:

1. **Calendar & Seasonality (9):** `year`, `quarter`, `month`, `week_of_year`, `is_q4`, `sin_week`, `cos_week`, `sin_month`, `cos_month`.
2. **Autoregressive Sales Lags (7):** `sales_lag_1`, `sales_lag_2`, `sales_lag_3`, `sales_lag_4`, `sales_lag_8`, `sales_lag_12`, `sales_lag_52`.
3. **Rolling Window Statistics (6):** `sales_rolling_mean_4`, `sales_rolling_std_4`, `sales_rolling_min_4`, `sales_rolling_max_4`, `sales_rolling_mean_12`, `sales_rolling_std_12` (all strictly shifted by 1 period before rolling calculations).
4. **Lagged Operational Drivers (4):** `orders_lag_1`, `quantity_lag_1`, `profit_lag_1`, `avg_discount_lag_1`.

---

## 9. Target Leakage Prevention & Auditing

- **No Contemporaneous Variables:** Unlagged operational metrics (same-week Profit, Quantity, Discount, Orders) were removed.
- **Shift Invariant:** For every row $t$, rolling features evaluate over $[t-k, t-1]$, strictly excluding week $t$.
- **Automated Leakage Audit (`src/leakage_check.py`):** Verified monotonic 7-day deltas, exact lag alignment, and absence of target leakage.

---

## 10. Baselines & Machine Learning Models

To prove genuine predictive learning, machine learning algorithms were benchmarked against historical persistence rules:
1. **Naive Baseline ($t-1$):** Assumes next week's revenue equals this week's revenue ($\hat{y}_t = y_{t-1}$).
2. **Seasonal Naive Baseline ($t-52$):** Assumes revenue equals the exact same week from the previous year ($\hat{y}_t = y_{t-52}$).
3. **Ridge Regression:** L2 regularized linear pipeline with `StandardScaler` ($\alpha=10.0$).
4. **Random Forest Regressor:** 100 trees, max depth 6, min samples split 4, min samples leaf 2.
5. **HistGradientBoostingRegressor:** 100 iterations, max depth 4, learning rate 0.05.
6. **Gradient Boosting Regressor:** 100 trees, max depth 3, learning rate 0.05.

---

## 11. Chronological Validation & Backtesting

Models were evaluated through expanding-window `TimeSeriesSplit` (5 folds) across the 104-week training period ($T_{\text{train, max}} < T_{\text{val, min}}$ across every fold):

| Model | Fold 1 MAE | Fold 2 MAE | Fold 3 MAE | Fold 4 MAE | Fold 5 MAE | **Mean MAE ($)** | **Mean RMSE ($)** | **Mean WAPE (%)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | $2,739.06 | $4,570.61 | $5,360.18 | $6,146.42 | $6,837.89 | **$5,130.83** | **$6,554.35** | 48.60% |
| **HistGradientBoosting** | $3,959.08 | $4,652.67 | $4,383.08 | $5,607.72 | $7,153.51 | **$5,151.23** | $6,778.72 | **46.52%** |
| **Random Forest** | $3,939.81 | $4,061.90 | $4,036.03 | $6,104.99 | $7,841.56 | **$5,196.86** | $6,602.18 | 47.12% |
| **Seasonal Naive Baseline** | $3,584.22 | $4,326.65 | $5,271.92 | $5,821.14 | $7,708.33 | **$5,342.45** | $6,864.83 | 49.14% |
| **Naive Baseline** | $2,732.68 | $5,645.55 | $5,815.11 | $5,475.29 | $7,163.05 | **$5,366.34** | $6,751.16 | 49.61% |
| **Gradient Boosting** | $5,491.56 | $4,809.91 | $5,014.63 | $5,324.96 | $8,027.09 | **$5,733.63** | $7,370.19 | 53.07% |

---

## 12. Final Model Selection

**Selected Final Model:** **Ridge Regression** (`StandardScaler` + `Ridge(alpha=10.0, random_state=42)`)

### Selection Rationale (`reports/final_model_selection.md`):
- **Empirical Generalization:** Lowest CV Mean MAE ($5,130.83) and lowest 2017 Holdout Test MAE ($5,377.97).
- **Sample Parsimony:** With only 104 training rows, L2 shrinkage prevents the noisy leaf-overfitting observed in deeper tree ensembles during volatile holiday periods.
- **Directional Transparency:** Linear coefficients directly indicate whether features drive or drag sales, facilitating executive trust.
- **Diagnostic Rigor:** Residuals are verified to be statistically unbiased ($p=0.158$), normally distributed ($p=0.330$), and free from serial autocorrelation ($d=2.06$).

---

## 13. Actual Performance Metrics

All metrics reflect genuine out-of-sample evaluation on the 53 weeks of the 2017 holdout test set:

| Model | Test MAE ($) | Test RMSE ($) | Test WAPE (%) | Test MAPE (%) | Beat Naive? | Beat Seasonal Naive? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline ($t-1$)** | $7,187.28 | $9,134.80 | 51.56% | 63.05% | Reference | Reference |
| **Seasonal Naive ($t-52$)** | $7,075.47 | $9,172.35 | 50.76% | 61.50% | Reference | Reference |
| **Ridge Regression (Selected)**| **$5,377.97** | **$6,778.95** | **38.58%** | **56.84%** | **YES (+25.17%)** | **YES (+23.99%)** |
| Gradient Boosting | $5,493.08 | $7,125.67 | 39.41% | 47.31% | YES (+23.57%) | YES (+22.36%) |
| Random Forest | $5,600.20 | $7,289.91 | 40.18% | 46.25% | YES (+22.08%) | YES (+20.85%) |
| HistGradientBoosting | $5,926.80 | $7,601.07 | 42.52% | 49.76% | YES (+17.54%) | YES (+16.23%) |

> [!NOTE]
> **Understanding WAPE:** Weighted Absolute Percentage Error ($\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$) weights errors by total sales volume, avoiding the distortion of traditional MAPE when weekly actuals are small. Ridge achieved **38.58% WAPE**, reducing baseline error by over 12 percentage points.

---

## 14. Error Analysis & Residual Diagnostics

Detailed investigation of 2017 holdout residuals (`reports/error_analysis.md`):
- **Global Bias Test (One-Sample t-test):** $t = -1.4339, p = 0.1576$. Fail to reject $H_0$; the model exhibits **no statistically significant bias**.
- **Normality Test (Shapiro-Wilk):** $W = 0.9751, p = 0.3302$. Fail to reject $H_0$; residuals conform to a normal distribution.
- **Autocorrelation (Durbin-Watson):** $d = 2.0597 \approx 2.0$. Zero first-order autocorrelation.
- **Asymmetric Peak vs. Trough Errors:**
  - *High-Sales Weeks (>75th percentile, >$19,484):* Systematically **underpredicted** by an average of **+$5,809.98/week** (peak miss).
  - *Low-Sales Weeks (<25th percentile, <$7,782):* Systematically **overpredicted** by an average of **-$7,078.89/week** (lag inertia miss).
  - *Top Outlier Week (2017-11-19, Black Friday):* Actual sales peaked at $35,344.42 vs forecast $19,618.78 (underprediction of $15,725.64).

---

## 15. Strategic Business Insights

The project synthesizes operational findings using the **OBSERVATION / INTERPRETATION / RECOMMENDATION** framework (`reports/business_insights.md`):

1. **Hard Margin Governance:**
   - *Observation:* Discounts $> 20\%$ trigger catastrophic losses (-12.4% to -84.2%), losing $156k in margin across 1,870 orders.
   - *Recommendation:* Enforce a hard approval cap on commercial discounts exceeding 20% and tie sales compensation to gross profit contribution.
2. **Restructure Furniture Merchandising:**
   - *Observation:* Tables lost -$17.7k and Bookcases lost -$3.5k due to heavy parcel freight and unbundled discounting.
   - *Recommendation:* Eliminate standalone discounting on Tables; bundle Furniture with high-margin Technology peripherals (Copiers, Displays).
3. **Data-Supported Inventory Planning:**
   - *Observation:* Peak holiday weeks underpredict demand by ~22.6% on average, while post-holiday slumps overpredict demand.
   - *Recommendation:* One possible approach would be to use forecast uncertainty/error distributions to determine an appropriate safety margin, accounting for empirical peak underprediction rather than assuming a flat static buffer.

---

## 16. Streamlit Interactive Dashboard

The interactive web application (`dashboard/app.py`) provides 6 multi-functional views:
- **Executive Overview:** Real-time KPI cards ($2.30M revenue, $286k profit, 12.5% margin), annual growth bars, and regional contribution.
- **Sales Analytics:** Multi-dimensional filters (Year, Category, Segment, Region), monthly timelines, sub-category profit bars, and state volume rankings.
- **Forecasting Engine:** Interactive 1-to-12 week horizon projection slider with sequential recursive lag propagation, historical 2017 actuals comparison, and CSV export.
- **Model Performance:** Side-by-side benchmark comparison tables and 5-fold backtesting metrics.
- **Strategic Insights:** Structured business playbooks covering revenue, seasonality, margin cliffs, and operational use cases.
- **About / Limitations:** Transparent data lineage, technical specifications, and governance disclaimers.

---

## 17. Repository Structure

```
Retail-Sales-Forecasting/
├── .streamlit/
│   └── config.toml                # Headless server & custom UI theme
├── assets/
│   ├── eda/                       # EDA distribution and correlation plots
│   ├── features/                  # Autocorrelation and feature heatmaps
│   └── models/                    # Model evaluation, backtest, and error plots
│       └── error_analysis/        # Residual diagnostic figures
├── dashboard/
│   ├── app.py                     # Main Streamlit web application
│   ├── data_utils.py              # Cached loaders & recursive forecasting engine
│   └── README.md                  # Dashboard documentation & launch instructions
├── data/
│   ├── raw/
│   │   └── Sample - Superstore.csv # Immutable raw dataset (SHA-256 verified)
│   └── processed/
│       ├── superstore_cleaned.csv  # Cleaned transaction records (9,993 rows)
│       └── forecasting_features.csv# 26-feature weekly dataset (157 rows)
├── models/
│   ├── final_model.joblib         # Production Ridge pipeline (trained on full data)
│   ├── ridge_regression.joblib    # Evaluation Ridge pipeline (trained on 2015-2016)
│   ├── random_forest.joblib       # Benchmark Random Forest pipeline
│   ├── gradient_boosting.joblib   # Benchmark Gradient Boosting pipeline
│   ├── histgradientboosting.joblib# Benchmark HistGradientBoosting pipeline
│   └── model_metadata.json        # Formal metadata, schema, and metrics
├── notebooks/
│   ├── 02_data_cleaning.ipynb     # Reproducible cleaning notebook
│   └── 03_eda.ipynb               # Exploratory data analysis notebook
├── reports/
│   ├── baseline_results.md        # Naive vs Seasonal Naive benchmarks
│   ├── model_comparison.md        # Comprehensive ML comparison report
│   ├── backtesting_results.md     # 5-fold expanding-window backtesting
│   ├── error_analysis.md          # Residual diagnostics & tier breakdown
│   ├── forecasting_limitations.md # Transparent assessment of constraints
│   ├── final_model_selection.md   # Multi-dimensional selection justification
│   ├── business_insights.md       # Strategic commercial playbooks
│   ├── model_card.md              # Production model card
│   └── deployment.md              # Local, Docker, and Cloud deployment guide
├── src/
│   ├── __init__.py
│   ├── config.py                  # Project-relative filesystem paths
│   ├── data_loader.py             # Raw data ingestion utilities
│   ├── validate_data.py           # Ingestion integrity & schema validation
│   ├── data_cleaning.py           # Data transformation pipeline
│   ├── analytics.py               # Aggregation & KPI calculation
│   ├── feature_engineering.py     # Leakage-free feature generator
│   ├── leakage_check.py           # Automated target leakage auditor
│   ├── modeling.py                # Model training, CV & evaluation engine
│   ├── backtesting.py             # Expanding-window backtesting engine
│   ├── error_analysis.py          # Residual diagnostics & worst-forecast extractor
│   └── forecasting.py             # Production model wrapper & export generator
├── tests/
│   ├── test_analytics.py          # KPI & aggregation unit tests
│   ├── test_backtesting.py        # Expanding backtesting invariant tests
│   ├── test_dashboard.py          # Dashboard loading, schema & KPI tests
│   ├── test_data_cleaning.py      # Deduplication & schema tests
│   ├── test_data_loader.py        # Ingestion & date parsing tests
│   ├── test_data_validation.py    # Raw hash & integrity tests
│   ├── test_deployment.py        # Dockerfile, ignore & secret tests
│   ├── test_error_analysis.py     # Residual math & tier partition tests
│   ├── test_feature_engineering.py# Temporal shift & feature schema tests
│   ├── test_forecasting.py        # Production inference & metadata tests
│   └── test_modeling.py           # Model instantiation & CV tests
├── .dockerignore                  # Docker build context exclusions
├── .env.example                   # Environment variable template
├── .gitignore                     # Git tracking exclusions
├── Dockerfile                     # Production multi-stage container spec
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # Minimal verified dependencies
```

---

## 18. Installation & Setup

### Prerequisites
- Python 3.11.x
- Git

### Setup Virtual Environment (Windows PowerShell)
```powershell
# Clone the repository
git clone https://github.com/your-username/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Upgrade pip and install verified dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 19. Running Locally

### Execute Complete ML Pipeline & Leakage Audit
```powershell
# Run data cleaning
python -m src.data_cleaning

# Generate leakage-free features
python -m src.feature_engineering

# Run automated leakage audit
python -m src.leakage_check

# Run modeling and evaluation
python -m src.modeling

# Run expanding-window backtesting
python -m src.backtesting

# Run error analysis and residual diagnostics
python -m src.error_analysis

# Train production model and export forecasts
python -m src.forecasting
```

### Launch Interactive Dashboard
```powershell
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

---

## 20. Docker Containerization

A production-ready `Dockerfile` and `.dockerignore` are provided:
- Base: `python:3.11-slim`
- Security: Runs under unprivileged `appuser` (UID 1000)
- Health Check: Native polling of `http://localhost:8501/_stcore/health`
- Port: `8501`

### Build and Run Commands
```bash
# Build the Docker image
docker build -t retail-sales-forecasting .

# Run the container locally
docker run --rm -p 8501:8501 retail-sales-forecasting
```

> [!NOTE]
> **Host Environment Status:**
> Docker configuration was created and validated structurally via automated unit tests (`tests/test_deployment.py`). Docker execution was not locally verified in this development session because the Docker Engine/CLI was unavailable on the local workstation.

---

## 21. Cloud Deployment Guide

The repository is configured for one-click deployment to **Streamlit Community Cloud**:

### Deployment Steps:
1. Push repository changes to GitHub:
   ```bash
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"** $\rightarrow$ select repository `Retail-Sales-Forecasting`, branch `main`, and main file path `dashboard/app.py`.
4. Click **"Deploy!"**. Streamlit Cloud will install dependencies from `requirements.txt` and serve the dashboard.

> [!NOTE]
> Cloud deployment was prepared but not executed in this development session because interactive third-party authentication (GitHub OAuth) is required.

---

## 22. Automated Test Suite

The project includes an exhaustive suite of **80 unit and integration tests** ensuring data immutability, mathematical invariants, temporal ordering, and deployment readiness:

```powershell
# Run the complete test suite
python -m pytest -v
```

```
============================= test session starts =============================
collected 80 items

tests/test_analytics.py (6 passed)
tests/test_backtesting.py (6 passed)
tests/test_dashboard.py (8 passed)
tests/test_data_cleaning.py (8 passed)
tests/test_data_loader.py (6 passed)
tests/test_data_validation.py (6 passed)
tests/test_deployment.py (7 passed)
tests/test_error_analysis.py (7 passed)
tests/test_feature_engineering.py (8 passed)
tests/test_forecasting.py (8 passed)
tests/test_modeling.py (10 passed)

============================= 80 passed in 10.72s =============================
```

---

## 23. Methodological Limitations

In the interest of scientific honesty and interview defensibility (`reports/forecasting_limitations.md`):
1. **Small Sample Horizon:** Only 4 calendar years exist; requiring a 52-week lag history leaves exactly **157 usable weekly periods** (2015–2017).
2. **Weekly Aggregation Level:** Intra-week daily volatility, weekend shopping spikes, and daily stockout risks are smoothed out.
3. **Supervised 1-Step-Ahead Formulation:** The native architecture predicts 1 week forward. Multi-step horizons (2–12 weeks) require recursive self-feeding of predicted lags, which compounds uncertainty over longer horizons.
4. **Absence of Exogenous Covariates:** The dataset lacks promotional campaign schedules, coupon marketing calendars, stockout/inventory records, weather data, competitor pricing, and macroeconomic indices.
5. **Forecast Error Magnitude:** Holdout WAPE is 38.58% (MAE: $5,377.97). While beating naive persistence by 25.2%, an average error of ±$5,300 means forecasts must be treated as directional guidance rather than deterministic truth.
6. **Future Performance:** Past out-of-sample backtest results do not guarantee future accuracy under shifting economic conditions.

---

## 24. Future Enhancements

- **Direct Multi-Horizon Modeling:** Implement independent direct models ($h=1, 2, \dots, 12$) to avoid recursive lag error compounding.
- **External Covariates:** Ingest national holiday calendars (floating dates like Thanksgiving/Labor Day) and macroeconomic consumer price indices.
- **Hierarchical Reconciliation:** Implement bottom-up category-level forecasting reconciled using MinT (Minimum Trace) reconciliation.
- **Probabilistic Forecasting:** Implement quantile regression or conformal prediction to output calibrated prediction intervals ($P_{10}, P_{50}, P_{90}$).

---

## 25. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
