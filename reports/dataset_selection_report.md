# Dataset Selection Report

## Project Objective
The primary objective of the **Retail Sales Forecasting & Analytics** project is to build an end-to-end, reproducible Data Science and Machine Learning system. The workflow entails:
1. Ingesting and validating real-world retail sales transactions.
2. Conducting in-depth exploratory data analysis (EDA) to discover business patterns across products, customer segments, and regions.
3. Engineering time-series features (lags, rolling statistics, calendar seasonality, discount interactions) using strict temporal ordering to prevent data leakage.
4. Developing, evaluating, and comparing statistical and machine learning forecasting models under rolling-origin / walk-forward validation.
5. Performing rigorous error analysis and delivering actionable business insights.
6. Deploying an interactive Streamlit dashboard and packaging the project for GitHub and Docker.

To fulfill these objectives without synthetic data or artificial assumptions, the underlying dataset must possess rich transaction-level attributes, a dependable date structure spanning multiple seasonal cycles, and public-friendly licensing.

---

## Candidate Datasets

The following four serious public retail datasets were researched and evaluated across all 21 technical and business dimensions:

| Evaluation Dimension | Candidate 1: Sample Superstore (Selected) | Candidate 2: UCI Online Retail II | Candidate 3: Walmart Store Sales Forecasting | Candidate 4: Rossmann Store Sales |
| :--- | :--- | :--- | :--- | :--- |
| **1. Dataset Name** | Sample - Superstore Retail Dataset | Online Retail II Dataset | Walmart Recruiting: Store Sales Forecasting | Rossmann Store Sales |
| **2. Original Source** | Tableau Software / Kaggle Open Data | UCI Machine Learning Repository | Kaggle / Walmart Inc. | Kaggle / Rossmann |
| **3. Direct Source URL** | [Kaggle / Tableau Mirror](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) | [UCI Dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | [Kaggle Walmart Competition](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) | [Kaggle Rossmann Competition](https://www.kaggle.com/c/rossmann-store-sales/data) |
| **4. License / Usage Terms** | CC0: Public Domain / Open Educational Use | Creative Commons Attribution 4.0 (CC BY 4.0) | Kaggle Competition Rules (Restricted Redistribution) | Open Database License (ODbL) |
| **5. Number of Records** | 9,994 rows (verified) | 1,067,371 rows (2-year version) | 421,570 rows | 1,017,209 rows (`train.csv`) |
| **6. Number of Columns** | 21 columns (verified) | 8 columns | 16 merged columns across 3 tables | 19 merged columns across 2 tables |
| **7. Date Range** | 2014-01-03 to 2017-12-30 (4 full calendar years) | 2009-12-01 to 2011-12-09 (2 years) | 2010-02-05 to 2012-10-26 (2.7 years) | 2013-01-01 to 2015-07-31 (2.5 years) |
| **8. Date/Time Column** | `Order Date`, `Ship Date` | `InvoiceDate` (timestamp) | `Date` (weekly) | `Date` (daily) |
| **9. Sales/Revenue Target** | `Sales` ($0.44 to $22,638.48; Total: $2,297,200.86) | Derived (`Quantity * UnitPrice`) | `Weekly_Sales` | `Sales` |
| **10. Quantity Field** | `Quantity` (1 to 14 units per transaction) | `Quantity` | None (revenue only) | None (customer count instead) |
| **11. Product/Category Fields** | `Category` (3), `Sub-Category` (17), `Product ID`, `Product Name` | `StockCode`, `Description` (no formal categories) | `Dept` (department number 1–99, anonymized) | None (store aggregate; no product categories) |
| **12. Customer/Store/Region** | `Customer ID`, `Segment` (3), `Region` (4), `State` (49), `City` (531) | `CustomerID` (5-digit ID), `Country` | `Store` (1–45), `Type` (A/B/C), `Size` (no customer info) | `Store` (1–1115), `StoreType`, `Assortment` |
| **13. Missing-Value Situation** | 0 missing values across all 21 columns (verified) | ~22% missing `CustomerID`, ~0.4% missing `Description` | >60% missing in `MarkDown1-5` promotional columns | Missing values in competition distance/open dates |
| **14. Data Granularity** | Order line item (individual item in order) | Invoice line item | Weekly department aggregate | Daily store aggregate |
| **15. Forecasting Suitability** | **High**: 4 full consecutive years (208 weeks, 48 months, 1,237 active order dates); enables robust daily/weekly/monthly forecasting with multi-year seasonal cycles | **Moderate**: 2 years of daily data; strong Christmas spikes, but only ~104 weekly points; irregular e-commerce days | **High**: 143 weekly observations across 45 stores; panel series | **High**: 942 daily observations across 1,115 stores; panel series |
| **16. EDA Suitability** | **Exceptional**: Multi-dimensional hierarchies (Product, Category, Sub-Category, Region, State, Segment, Discount, Margin, Shipping Time) | **Moderate**: Strong for RFM and text description, but lacks product taxonomy and geography | **Low**: Anonymized stores, departments, and dates; lacks customer and product narrative | **Moderate**: Good for promotions, school holidays, and store types, but lacks product categories |
| **17. ML Suitability** | **High**: Both regression/classification at line-item level and time-series modeling (ARIMA, Prophet, Random Forest, XGBoost) | **Moderate**: Great for clustering/RFM; requires aggregation for time-series | **High**: Well-suited for panel gradient boosting | **High**: Well-suited for panel gradient boosting |
| **18. Laptop Practicality** | **Ideal**: ~2 MB file; loads instantaneously, zero memory strain, rapid test execution and container builds | **Moderate**: ~80 MB file; ~1M rows; heavier in memory (~300MB RAM) | **Moderate**: Multi-table joins; ~420k rows | **Heavy**: >1M rows; training 1,115 store models is slow on a local laptop |
| **19. Data-Quality Problems** | Varying order cadence per day (demands calendar resampling/zero-imputation for daily time-series); outliers in discounts | Negative quantities (returns/cancellations); postage fees entered as items; zero unit prices | Massive missing values in Markdowns; negative sales entries | Closed store days with 0 sales require two-stage modeling; missing competitor dates |
| **20. Data-Leakage Risks** | `Ship Date` and `Profit` must not be used as features to forecast `Sales` at order time | Cancellations appearing later in time must not be leaked into past customer features | Markdown data starts late (Nov 2011), causing temporal leakage if not handled | `Customers` is contemporaneous (unknown at prediction time); using it leaks the target |
| **21. Public GitHub Suitability** | **100% Compliant**: CC0 public domain, lightweight, commits cleanly without Git LFS | **Compliant**: CC BY 4.0; large file size requires Git LFS | **Non-Compliant**: Restricted Kaggle competition terms forbid public rehosting | **Borderline**: ODbL allows sharing, but 100MB+ uncompressed size exceeds GitHub limits |

---

## Final Dataset
**Selected:** **Sample - Superstore Retail Dataset**

### Justification for Selection:
1. **Four Full Years of Continuous Temporal Depth:** Unlike datasets covering only 1 or 2 years, Superstore provides exactly 4 continuous calendar years (2014 through 2017). This provides 4 complete annual seasonal cycles (208 weeks / 48 months / 1,458 calendar days), making genuine out-of-time walk-forward validation possible without artificial data splitting.
2. **Transaction-Level Granularity with Real Business Attributes:** It provides both transaction-level granularity (9,994 line items across 5,009 distinct orders) and rich business dimensions: 3 Segments, 3 Categories, 17 Sub-Categories, 4 Regions, 49 States, and 531 Cities.
3. **Simultaneous Revenue, Unit, Margin, and Pricing Fields:** It tracks `Sales`, `Quantity`, `Discount`, and `Profit`. This enables multi-faceted EDA (e.g., price elasticity, discount vs. profit margin destruction, regional profitability) that cannot be performed on volume-only or revenue-only datasets.
4. **Clean, Verified Data Free from Synthetics:** Direct inspection of the raw CSV confirms zero missing values across all core attributes, genuine variance across products and orders, and a coherent date hierarchy (`Order Date` strictly precedes or equals `Ship Date`).
5. **Practicality and Reproducibility:** At ~2 MB, it is lightweight, runs tests in seconds, builds effortlessly in Docker, and can be committed directly to a public GitHub portfolio without cumbersome external storage dependencies or Git LFS.
6. **Clear Licensing:** It is licensed under CC0 / Public Domain / Open Educational Use, ensuring full compliance for an open-source GitHub portfolio and technical interviews.

---

## Data Characteristics
Verified directly against the raw dataset loaded into `data/raw/Sample - Superstore.csv`:

* **Total Records (Rows):** 9,994
* **Total Columns:** 21
* **Date Range:** 
  * Earliest Order Date: `2014-01-03`
  * Latest Order Date: `2017-12-30`
  * Total Days in Span: 1,457 days (~4.0 years)
  * Unique Order Dates: 1,237 dates
* **Annual Breakdown of Transactions:**
  * 2014: 1,993 records
  * 2015: 2,102 records
  * 2016: 2,587 records
  * 2017: 3,312 records
* **Total Sales Volume:** $2,297,200.86
* **Total Profit:** $286,397.02
* **Granularity:** Order-item line level (one record per product per order).

### Core Fields & Types
| Field Name | Type | Verified Domain / Range | Description |
| :--- | :--- | :--- | :--- |
| `Row ID` | Integer | 1 to 9,994 | Unique record identifier |
| `Order ID` | String | 5,009 unique IDs | Unique customer purchase order ID |
| `Order Date` | Date | 2014-01-03 to 2017-12-30 | Date when order was placed |
| `Ship Date` | Date | 2014-01-07 to 2018-01-05 | Date when order was shipped |
| `Ship Mode` | Categorical | 4 values (Standard Class, Second Class, First Class, Same Day) | Shipping method selected |
| `Customer ID` | String | 793 unique customers | Unique customer identifier |
| `Customer Name` | String | 793 unique names | Name of customer |
| `Segment` | Categorical | 3 values (Consumer: 5,191, Corporate: 3,020, Home Office: 1,783) | Customer business segment |
| `Country` | Categorical | 1 value (United States) | Country of sale |
| `City` | Categorical | 531 cities | City of delivery |
| `State` | Categorical | 49 states | State of delivery |
| `Postal Code` | Numeric/String| 5-digit zip code | Delivery zip code |
| `Region` | Categorical | 4 values (West: 3,203, East: 2,848, Central: 2,323, South: 1,620) | Geographic region |
| `Product ID` | String | 1,862 unique IDs | Unique product code |
| `Category` | Categorical | 3 values (Office Supplies: 6,026, Furniture: 2,121, Technology: 1,847) | Broad product category |
| `Sub-Category` | Categorical | 17 sub-categories | Specific product family |
| `Product Name` | String | 1,850 unique names | Detailed catalog product description |
| `Sales` | Float | $0.44 to $22,638.48 | Transaction revenue (in USD) |
| `Quantity` | Integer | 1 to 14 units | Units sold |
| `Discount` | Float | 0.0 to 0.8 (0% to 80%) | Discount applied to transaction |
| `Profit` | Float | -$6,599.98 to $8,399.98 | Net transaction profit/loss |

---

## License and Usage
* **License:** Creative Commons CC0: Public Domain / Open Educational Use.
* **Commercial / Portfolio Rights:** Fully open for academic, portfolio, personal, and educational showcase.
* **Redistribution Rights:** Fully permissible to store directly in a public GitHub repository.

---

## Forecasting Suitability
1. **Four Full Calendar Years (48 Months / 208 Weeks):**
   * Forecasting retail sales requires observing the same seasonal spikes multiple times (specifically the Q4 holiday surge in September–December). Superstore contains 4 complete iterations of this cycle.
2. **Flexible Temporal Aggregation:**
   * **Daily Level:** 1,237 active order dates out of 1,458 calendar days. Can be regularized via continuous calendar resampling (filling non-trading days with 0 sales) to evaluate daily demand forecasting.
   * **Weekly Level:** 208 continuous, contiguous weekly observations (`W-SUN` or `W-MON`). This is ideal for medium-term operational retail forecasting with virtually no zero-demand gaps.
   * **Monthly Level:** 48 contiguous monthly observations, suitable for macro trend and strategic revenue forecasting.
3. **Multi-Series Forecasting Opportunities:**
   * Beyond total company sales, the data supports forecasting across the 3 product categories (`Technology`, `Furniture`, `Office Supplies`) and the 4 geographic regions (`West`, `East`, `Central`, `South`), enabling hierarchical time-series analysis.
4. **Time-Series-Safe Validation Design:**
   * Training set: 2014-01-03 to 2016-12-31 (3 full years: 75% of data, 156 weeks).
   * Test set: 2017-01-01 to 2017-12-30 (1 full out-of-time evaluation year: 25% of data, 52 weeks).
   * Rolling-origin cross-validation: Expanding windows over 2015 and 2016 to prevent any look-ahead leakage.

---

## Expected Challenges
We clearly distinguish between verified facts from our raw inspection and assumptions/challenges to be tackled during data cleaning and feature engineering:

### Verified Facts
1. **Irregular Daily Cadence:** Transactions do not occur on every single calendar day (1,237 unique dates out of 1,457 days in the span). A daily time series requires resampling to a complete `D` calendar index and filling missing days with zero sales.
2. **Extreme Value Skew:** Sales exhibits a strong right skew (minimum $0.44, median $54.49, maximum $22,638.48). Extreme sales outliers in high-value technology items will require log transformations or robust scaling during modeling.
3. **Negative Profits:** 1,871 transactions (18.7% of all rows) have negative profit (up to -$6,599.98) driven by heavy discounting (discounts above 20%).
4. **Strict Temporal Integrity:** `Ship Date` is always greater than or equal to `Order Date` (shipping duration ranges from 0 to 7 days).

### Assumptions & Implementation Considerations
1. **Zero-Demand Imputation:** Imputing 0 for non-trading days in daily series accurately reflects retail store operation, but weekly aggregation provides a smoother, more continuous target distribution for machine learning.
2. **Data-Leakage Guardrails:** Features generated for forecasting sales on a given day/week $t$ must strictly rely on information known prior to $t$ (e.g., lag sales $t-1, t-2, t-7, t-14, t-30$, rolling means over past windows). `Ship Date` and `Profit` must be strictly excluded from the feature matrix when forecasting sales demand at order time.

---

## Source
* **Dataset Name:** Sample - Superstore Retail Dataset
* **Original Publisher:** Tableau Software (Sample Data for Business Analytics & Visualization)
* **Canonical Open Data Source:** [Tableau Public Sample Data](https://community.tableau.com/s/sample-data) / [Kaggle Dataset Mirror](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
* **Raw Verified File Location:** `data/raw/Sample - Superstore.csv`
