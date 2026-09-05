# Exploratory Data Analysis Report

**Project:** Retail Sales Forecasting & Analytics  
**Phase:** Phase 6 – Exploratory Data Analysis (EDA)  
**Data Source:** `data/processed/superstore_cleaned.csv` (9,993 transactions, 21 columns)  
**Status:** Completed & Validated  

---

## Dataset Overview

The analyzed dataset comprises retail transactional sales records from the Superstore retail catalog spanning four complete consecutive calendar years (January 3, 2014 to December 30, 2017).

### Verified Dimensions and Cardinality
- **Total Records / Transactions:** 9,993
- **Unique Orders:** 5,009 distinct purchase orders (averaging ~2.0 transactions per order)
- **Unique Customers:** 793 distinct enterprise and individual clients
- **Unique Products:** 1,862 distinct items
- **Product Categories:** 3 broad categories (`Technology`, `Furniture`, `Office Supplies`)
- **Product Sub-Categories:** 17 distinct sub-categories
- **Geographic Coverage:** 4 Regions, 49 US States, and 531 Cities
- **Temporal Horizon:** 1,457 calendar days, with commercial order activity observed on 1,237 unique dates (84.9% active calendar days)

> [!NOTE]
> **Observation vs. Order Invariant:** Each row represents an individual line item on a purchase order, not an independent order. A single order frequently encompasses multiple products with independent pricing, discounts, and margins.

---

## Key KPIs

The table below summarizes high-level operational and financial KPIs calculated from the cleaned dataset:

| Metric | Formula / Definition | Verified Dataset Value |
| :--- | :--- | :--- |
| **Total Revenue (Sales)** | $\sum \text{Sales}$ | **$2,296,919.49** |
| **Total Net Profit** | $\sum \text{Profit}$ | **$286,409.08** |
| **Total Quantity Sold** | $\sum \text{Quantity}$ | **37,871 units** |
| **Unique Orders** | $\text{CountDistinct}(\text{Order ID})$ | **5,009 orders** |
| **Unique Customers** | $\text{CountDistinct}(\text{Customer ID})$ | **793 customers** |
| **Average Order Value (AOV)** | $\frac{\text{Total Sales}}{\text{Unique Orders}}$ | **$458.56 per order** |
| **Average Profit per Order** | $\frac{\text{Total Profit}}{\text{Unique Orders}}$ | **$57.18 per order** |
| **Overall Profit Margin** | $\frac{\text{Total Profit}}{\text{Total Sales}} \times 100$ | **12.47%** |
| **Average Items per Order** | $\frac{\text{Total Quantity}}{\text{Unique Orders}}$ | **7.56 units per order** |

---

## Sales Trends

The business experienced continuous multi-year top-line expansion between 2014 and 2017, accompanied by annual cyclicality.

### Annual Breakdown & Growth Rates
| Year | Annual Sales ($) | Annual Profit ($) | Orders | Units Sold | Profit Margin (%) | YoY Sales Growth (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2014** | $483,966.13 | $49,556.03 | 968 | 7,579 | 10.24% | Baseline |
| **2015** | $470,532.51 | $61,618.60 | 1,037 | 7,979 | 13.10% | -2.78% |
| **2016** | $609,205.60 | $81,795.17 | 1,315 | 9,848 | 13.43% | +29.47% |
| **2017** | $733,215.26 | $93,439.27 | 1,689 | 12,465 | 12.74% | +20.36% |

- **Observation:** Revenue was stable between 2014 and 2015 (-2.78%), followed by rapid acceleration in 2016 (+29.47%) and 2017 (+20.36%).
- **Interpretation:** The business scaled order volume from 968 to 1,689 annual orders (+74.5%) while maintaining a healthy net profit margin between 10.2% and 13.4%.
- **Visualization:** See [monthly_sales_trend.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/monthly_sales_trend.png).

---

## Seasonality

### 1. Monthly Seasonality (The Q4 Peak)
Monthly revenue aggregation reveals a pronounced and recurring annual seasonal wave:

| Month | Total Sales ($) | Sales Share (%) | Profit ($) | Monthly Pattern Description |
| :--- | :--- | :--- | :--- | :--- |
| **January** | $94,924.84 | 4.13% | $9,134.45 | Post-holiday low; seasonal slump |
| **February** | $59,751.25 | 2.60% | $10,294.61 | Lowest sales volume of the calendar year |
| **March** | $205,005.49 | 8.93% | $28,594.69 | Q1 fiscal corporate replenishment surge |
| **April** | $137,480.76 | 5.99% | $11,599.50 | Moderate mid-spring baseline |
| **May** | $155,028.81 | 6.75% | $22,411.31 | Steady early-summer volume |
| **June** | $152,718.68 | 6.65% | $21,285.80 | Mid-year steady state |
| **July** | $147,238.10 | 6.41% | $13,832.66 | Summer low-variance baseline |
| **August** | $159,044.06 | 6.92% | $21,776.94 | Back-to-school ramp-up begins |
| **September**| $307,649.95 | 13.39% | $36,857.48 | Strong Q3 end; early corporate procurement |
| **October** | $200,322.98 | 8.72% | $31,784.04 | Pre-holiday interim demand |
| **November** | $352,461.07 | **15.34%** | $35,468.43 | Peak month; Black Friday / Cyber Monday surge |
| **December** | $325,293.50 | **14.16%** | $43,369.19 | Year-end corporate budget flush & holiday retail |

- **Key Evidence:** The final 4 months (September to December) generate **$1,185,727.50**, representing **51.62%** of all revenue.
- **Forecasting Implication:** Any predictive model must account for strong annual seasonality (e.g. month-of-year indicators or Fourier terms).

### 2. Day-of-Week Seasonality
- **Monday & Friday:** Lead weekly order volume (Monday: 18.67%, Friday: 18.62%).
- **Wednesday:** Unusually low volume (3.82% of sales, 370 transactions), representing a midweek ordering dip.
- **Weekend:** Saturday (15.59%) and Sunday (16.78%) show sustained consumer demand.
- **Visualization:** See [forecast_seasonality_patterns.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/forecast_seasonality_patterns.png).

---

## Profitability

While the company achieves an aggregate profit of $286,409.08 (12.47% margin), profitability is unevenly distributed:

### Profitable vs. Loss-Making Distribution
- **Profitable Transactions:** 8,058 rows (80.64%) generate $442,548.71 in gross profit.
- **Break-Even Transactions ($0.00 profit):** 65 rows (0.65%).
- **Loss-Making Transactions:** 1,870 rows (18.71%) generate **-$156,139.63** in cumulative losses.
- **Maximum Loss Single Item:** -$6,599.98 (Row ID 7773: 3D Systems CubePro Trio 3D Printer sold at a 70% discount).
- **Core Driver of Losses:** The average discount for loss-making items is **37.8%**, compared to only **10.5%** for profitable items.

---

## Product Performance

### Product Sales Concentration
- The top 50 products (out of 1,862, representing 2.7% of catalog) generate **$754,233.15** (32.8% of total revenue).
- The catalog exhibits a classic long-tail retail distribution.

### Top 5 Products by Sales
1. `Canon imageCLASS 2200 Advanced Copier`: $61,599.82 Sales (8 orders, $25,199.93 Profit)
2. `Fellowes PB500 Electric Punch Plastic Comb Binding Machine`: $27,453.38 Sales (10 orders, $7,753.04 Profit)
3. `Cisco TelePresence System EX90 Videoconferencing Unit`: $22,638.48 Sales (1 order, -$1,811.08 Loss)
4. `Hewlett Packard LaserJet 3310 Copier`: $18,839.69 Sales (8 orders, $6,983.88 Profit)
5. `GBC DocuBind P400 Electric Binding System`: $17,965.07 Sales (6 orders, -$5,496.39 Loss)

### Top 3 Most Unprofitable Products
1. `Cubify CubeX 3D Printer Double Head Print`: -$8,879.97 Cumulative Loss across 3 sales (average discount: 60%).
2. `Lexmark MX611dhe Monochrome Laser Printer`: -$4,589.97 Cumulative Loss across 4 sales.
3. `Cubify CubeX 3D Printer Triple Head Print`: -$3,839.99 Cumulative Loss.

---

## Category Performance

Performance diverges sharply across the 3 product categories:

| Category | Total Sales ($) | Sales Share (%) | Net Profit ($) | Profit Margin (%) | Units Sold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Technology** | $836,154.03 | 36.40% | $145,454.95 | **17.40%** | 6,939 |
| **Office Supplies** | $719,047.03 | 31.30% | $122,490.80 | **17.04%** | 22,906 |
| **Furniture** | $741,718.42 | 32.29% | $18,463.33 | **2.49%** | 8,026 |

- **Observation:** Furniture accounts for almost a third of total revenue ($741.7k) but produces a negligible net profit margin (2.49%), generating less than an eighth of the profit produced by Technology.
- **Visualization:** See [category_sales_profit.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/category_sales_profit.png).

### Sub-Category Breakdown & Loss-Leader Sub-Categories
Examining the 17 sub-categories reveals the source of Furniture's margin erosion:
- **`Tables` (Furniture):** $206,965.53 Sales, **-$17,725.48 Profit** (-8.56% margin). Average discount is 26.1%.
- **`Bookcases` (Furniture):** $114,880.00 Sales, **-$3,472.56 Profit** (-3.02% margin). Average discount is 21.1%.
- **`Supplies` (Office Supplies):** $46,673.54 Sales, **-$1,189.10 Profit** (-2.55% margin).
- **`Copiers` (Technology):** Highly lucrative with **$55,617.82 Profit** on $149,528.03 Sales (**37.20% margin**).
- **`Paper` (Office Supplies):** High-volume staple with **$34,053.57 Profit** on $78,479.21 Sales (**43.39% margin**).
- **Visualization:** See [subcategory_profit_margin.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/subcategory_profit_margin.png).

---

## Customer Segments

| Segment | Sales ($) | Sales Share (%) | Profit ($) | Margin (%) | Orders | Average Order Value (AOV) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer** | $1,161,401.34 | 50.56% | $134,119.21 | 11.55% | 2,586 | $449.11 |
| **Corporate** | $706,146.36 | 30.74% | $91,979.13 | 13.03% | 1,514 | $466.41 |
| **Home Office** | $429,371.79 | 18.69% | $60,310.74 | 14.05% | 909 | $472.36 |

- **Observation:** The Consumer segment generates slightly over half of total revenue. However, Home Office customers demonstrate the highest AOV ($472.36) and the highest profit margin (14.05%).

---

## Geographic Performance

### Regional Summary
| Region | Sales ($) | Profit ($) | Profit Margin (%) | Unique Orders | Units Sold |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **West** | $725,457.82 | $108,418.45 | **14.94%** | 1,611 | 12,266 |
| **East** | $678,499.87 | $91,534.84 | **13.49%** | 1,401 | 10,616 |
| **Central** | $501,239.89 | $39,706.36 | **7.92%** | 1,175 | 8,780 |
| **South** | $391,721.91 | $46,749.43 | **11.93%** | 822 | 6,209 |

- **Observation:** The West and East regions together generate 61.1% of revenue and 69.8% of net profit. The Central region exhibits the weakest profit margin (7.92%), despite achieving higher sales than the South.
- **Visualization:** See [regional_performance.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/regional_performance.png).

### State Profitability Extremes
- **Top 3 Profitable States:**
  1. California: $457,687.63 Sales, +$76,381.39 Profit (16.69% margin)
  2. New York: $310,876.27 Sales, +$74,038.55 Profit (23.82% margin)
  3. Washington: $138,641.27 Sales, +$33,402.65 Profit (24.09% margin)
- **Top 3 Unprofitable States (Significant Losses):**
  1. Texas: $170,188.05 Sales, **-$25,729.36 Loss** (-15.12% margin)
  2. Ohio: $77,976.76 Sales, **-$16,959.32 Loss** (-21.75% margin)
  3. Pennsylvania: $116,511.91 Sales, **-$15,559.96 Loss** (-13.35% margin)

---

## Shipping Analysis

| Ship Mode | Transactions | Sales ($) | Profit ($) | Profit Margin (%) | Mean Lead Time | Min / Max Lead Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Class** | 5,967 (59.7%) | $1,358,215.93 | $164,088.79 | 12.08% | 5.01 days | 3 to 7 days |
| **Second Class** | 1,945 (19.5%) | $459,193.57 | $57,446.64 | 12.51% | 3.24 days | 1 to 5 days |
| **First Class** | 1,538 (15.4%) | $351,428.42 | $48,969.84 | 13.93% | 2.18 days | 1 to 4 days |
| **Same Day** | 543 (5.4%) | $128,081.57 | $15,903.81 | 12.42% | 0.04 days | 0 to 1 days |

- **Observation:** Standard Class handles ~60% of all volume with a consistent 5-day lead time. Profit margins remain stable between 12.0% and 13.9% across all four shipping modes.

---

## Discount Analysis

Discounts exert a profound structural impact on transaction profitability:

| Discount Tier | Transactions | Total Sales ($) | Total Profit ($) | Profit Margin (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0% (No Discount)** | 4,798 (48.0%) | $1,087,908.27 | $320,987.60 | **+29.51%** | Highly Profitable |
| **0.01% – 20%** | 3,803 (38.1%) | $846,522.22 | $100,785.47 | **+11.91%** | Moderately Profitable |
| **20.01% – 40%** | 459 (4.6%) | $233,856.45 | -$35,805.41 | **-15.31%** | Loss-Making |
| **40.01% – 60%** | 215 (2.2%) | $71,048.21 | -$28,944.19 | **-40.74%** | Severe Loss |
| **60.01% – 80%** | 718 (7.2%) | $57,584.04 | -$70,614.40 | **-122.63%** | Extreme Loss |

- **Evidence-Based Finding:** At discounts of 20% or less, transactions maintain a positive aggregate profit margin (+11.9% to +29.5%). As soon as discounts exceed 20%, every single discount tier exhibits a negative net margin.
- **Correlation Caution:** We observe a strong empirical association ($r = -0.219$) between discount levels and profit. This reflects commercial pricing policies where deep promotions destroy product margins.
- **Visualization:** See [discount_vs_profit.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/discount_vs_profit.png).

---

## Correlation Analysis

Pearson correlation coefficients computed across quantitative variables:

| Variable Pair | Pearson $r$ | Statistical Interpretation |
| :--- | :--- | :--- |
| **Sales vs. Profit** | **+0.479** | Moderate positive linear correlation; higher sales generally yield higher absolute profit. |
| **Sales vs. Quantity** | **+0.201** | Weak positive correlation; large sales figures are driven more by unit price than by basket unit counts. |
| **Discount vs. Profit** | **-0.219** | Negative correlation; higher discount rates are associated with diminished transaction profitability. |
| **Discount vs. Sales** | **-0.028** | Negligible linear correlation; discounting does not substantially increase transaction dollar volume. |
| **Quantity vs. Profit** | **+0.066** | Negligible linear correlation. |

- **Visualization:** See [correlation_heatmap.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/correlation_heatmap.png).

---

## Outlier Analysis

- **Sales Distribution:** Heavily right-skewed (median $54.49, mean $229.86, maximum $22,638.48, standard deviation $623.25).
- **Log1p Transformation:** Applying $\ln(1 + \text{Sales})$ normalizes the distribution, demonstrating suitability for parametric regression modeling.
- **Top Extreme Sales Records:** Verified as genuine enterprise technology hardware purchases:
  - Row ID 2698: $22,638.48 (Cisco TelePresence System)
  - Row ID 6827: $17,499.95 (Canon imageCLASS Copier)
  - Row ID 8154: $13,999.96 (Canon imageCLASS Copier)
- **Conclusion:** These observations represent legitimate business sales and must not be trimmed or artificially clipped.
- **Visualization:** See [sales_distribution.png](file:///c:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/assets/eda/sales_distribution.png).

---

## Forecasting Implications

The exploratory findings establish crucial architectural parameters for Phase 7 (Feature Engineering) and Phase 8 (Model Development):

1. **Target Selection:** Total aggregate revenue (`Sales`) is the primary operational forecasting target. Secondary forecasting targets include Category-level sales (`Technology`, `Furniture`, `Office Supplies`).
2. **Granularity & Frequency:**
   - **Weekly Aggregation (`W-SUN`):** Highly recommended as the primary forecasting frequency. Spanning 208 continuous weeks across 4 full years provides a stable, zero-gap time series with clean multi-week seasonality.
   - **Daily Aggregation (`D`):** Requires regular calendar expansion with zero-demand imputation on non-active trading dates (1,237 active dates out of 1,457 calendar days).
3. **Temporal Invariant & Validation Split:**
   - Training Window: 2014-01-03 to 2016-12-31 (3 full years, 156 weeks, 75% of data).
   - Test Window: 2017-01-01 to 2017-12-30 (1 full year, 52 weeks, 25% of data).
   - Rolling Origin: Expanding window walk-forward validation across 2015 and 2016.
4. **Candidate Feature Families for Forecasting:**
   - **Calendar Seasonality:** Month of year, week of year, quarter, day of week.
   - **Autoregressive Lags:** Lagged sales at $t-1, t-2, t-4, t-52$ (capturing yearly seasonality).
   - **Rolling Statistics:** 4-week, 8-week, and 12-week moving averages and rolling standard deviations.
   - **Holiday Surge Flags:** Indicator variables for Q4 peak periods (September–December).

---

## Key Evidence-Based Findings

1. **Top-Line Growth:** Revenue grew from $484k (2014) to $733k (2017), an overall increase of 51.5%.
2. **Q4 Revenue Dependency:** Over 51.6% of annual revenue is concentrated in the final four months (September to December), driven by back-to-school and holiday procurement.
3. **Furniture Margin Vulnerability:** Furniture produces substantial revenue ($741.7k) but operates at an aggregate margin of only 2.49%, severely impaired by heavy discounting on Tables (-8.56% margin) and Bookcases (-3.02% margin).
4. **20% Discount Profitability Cliff:** Transactions with discounts $\le 20\%$ maintain positive profit margins. Transactions with discounts $> 20\%$ consistently produce net commercial losses.
5. **Geographic Concentration:** California and New York account for 33.5% of total sales and generate +$150,419.94 in net profit, while Texas, Ohio, and Pennsylvania account for cumulative losses of -$58,248.64 due to aggressive discounting.

---

## Limitations

1. **Lack of Inventory and Stock Data:** The dataset records fulfilled order transactions but does not track inventory stock-outs or unfulfilled customer demand.
2. **Lack of Marketing Spend Data:** Promotional intensity is measured via transaction discount percentages; external advertising spend or digital campaign data is not available.
3. **Calendar Resampling Necessity:** Daily order dates have intermittent non-trading days, necessitating weekly aggregation or zero-imputation during time-series modeling.
