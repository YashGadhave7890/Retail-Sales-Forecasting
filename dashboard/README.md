# Retail Sales Forecasting & Analytics — Interactive Dashboard

This directory contains the interactive Streamlit web application providing executive commercial analytics, slice-and-dice sales filtering, recursive multi-step forecasting, model benchmarking, and business strategy playbooks.

---

## 1. How to Launch the Dashboard

From the project root directory, launch the application using the project virtual environment:

```powershell
# Using the project's virtual environment:
.\.venv\Scripts\streamlit run dashboard/app.py
```

To run in headless mode (e.g. for testing or automated environments):

```powershell
.\.venv\Scripts\streamlit run dashboard/app.py --server.headless true --server.port 8501
```

Once launched, navigate to `http://localhost:8501` in your browser.

---

## 2. Required Artifacts & Files

The dashboard requires the following pre-computed project artifacts:

| Category | Artifact Path | Description |
| :--- | :--- | :--- |
| **Transaction Data** | `data/processed/superstore_cleaned.csv` | Cleaned transaction line items (9,993 rows) |
| **Forecasting Features** | `data/processed/forecasting_features.csv` | Weekly leakage-free feature dataset (157 rows) |
| **Production Model** | `models/final_model.joblib` | Serialized Ridge Regression pipeline |
| **Model Metadata** | `models/model_metadata.json` | Model parameters, metrics, and schema |
| **Model Comparisons** | `reports/model_comparison.csv` | Benchmark metrics against Naive, Random Forest, etc. |
| **Backtesting Data** | `reports/backtesting_results.csv` | Fold-by-fold results across 5 expanding windows |

---

## 3. Supported Functionality & Application Pages

### 1. Executive Overview (`Executive Overview`)
- High-level commercial KPIs calculated dynamically: Total Revenue ($2.30M), Total Profit ($286k), Margin (12.5%), Total Units (37,873), Orders (5,009), Average Order Value ($458.61).
- Annual growth progression and monthly seasonal curves.
- Category revenue and profit contribution breakdown.

### 2. Interactive Sales Analytics (`Sales Analytics`)
- Interactive multi-dimensional filtering by Order Year, Product Category, Customer Segment, and Geographic Region.
- Filtered KPIs and monthly time series.
- Sub-category profitability ranking highlighting loss-leaders (Tables, Bookcases).
- Top states by commercial volume.

### 3. Forecasting Engine (`Forecast`)
- **Future Horizon Projection:** Interactive slider for 1 to 12 weeks forward projection starting from the last Sunday (`2017-12-31`).
  - Week 1 uses observed historical lags.
  - Weeks 2–12 utilize sequential recursive lag propagation.
  - Point forecasts only (no synthetic confidence intervals).
  - Downloadable forecast CSV.
- **Historical Holdout Benchmark:** Compares model point predictions against actual sales for the 53 weeks of 2017 with Holdout Test MAE ($5,377.97) and WAPE (38.58%).

### 4. Model Performance (`Model Performance`)
- Side-by-side comparison table and bar charts comparing Ridge Regression against Naive, Seasonal Naive, Random Forest, HistGradientBoosting, and Gradient Boosting.
- 5-fold expanding-window backtesting robustness statistics.

### 5. Strategic Business Insights (`Business Insights`)
- Actionable executive playbooks covering Revenue Growth (+51.4%), Q4 Seasonality (34.2%), the 20% Discount Margin Cliff, Furniture Loss-Leaders, and Forecast Peak/Trough error distributions.
- Structured into **Observation**, **Interpretation**, and **Recommendation** tiers.

### 6. About & Limitations (`About / Limitations`)
- Transparent documentation of technical setup, data lineage, sample-size constraints, and governance disclaimers.

---

## 4. Known Methodological Limitations

1. **Small Sample Size:** 157 weekly periods (2015–2017) after reserving 52 weeks for annual seasonal lags.
2. **Weekly Level:** Daily intra-week shopping volatility and stockout events are smoothed out.
3. **1-Step-Ahead Nature:** The underlying model is trained on 1-step-ahead targets; multi-step projections use recursive self-feeding which compounds uncertainty over longer horizons.
4. **No Exogenous Features:** Promotional marketing schedules, coupon discounts, inventory stockouts, competitor pricing, and weather are not recorded in the dataset.
5. **Human-in-the-Loop:** Forecasts represent statistical directional guidance (WAPE ~38%) and must be combined with commercial operational intelligence prior to issuing purchase orders or making financial commitments.
