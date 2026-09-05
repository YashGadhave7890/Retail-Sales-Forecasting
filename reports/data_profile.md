# Data Profile Report

## 1. Dataset Dimensions
- **Total Rows:** 9,994
- **Total Columns:** 21
- **Total Cells:** 209,874
- **Memory Usage:** ~1.6 MB in memory

---

## 2. Schema and Column Metadata
All 21 columns are cataloged below with data type, unique value cardinality, missingness count, and verified sample value:

| # | Column Name | Raw Data Type | Parsed Type | Cardinality (Unique) | Missing Count | Missing % | Sample Value |
| :- | :--- | :--- | :--- | :- | :- | :- | :--- |
| 1 | `Row ID` | `int64` | `int64` | 9,994 | 0 | 0.0% | `1` |
| 2 | `Order ID` | `object` | `string` | 5,009 | 0 | 0.0% | `CA-2016-152156` |
| 3 | `Order Date` | `object` | `datetime64[ns]` | 1,237 | 0 | 0.0% | `2016-11-08` |
| 4 | `Ship Date` | `object` | `datetime64[ns]` | 1,334 | 0 | 0.0% | `2016-11-11` |
| 5 | `Ship Mode` | `object` | `string` | 4 | 0 | 0.0% | `Second Class` |
| 6 | `Customer ID` | `object` | `string` | 793 | 0 | 0.0% | `CG-12520` |
| 7 | `Customer Name` | `object` | `string` | 793 | 0 | 0.0% | `Claire Gute` |
| 8 | `Segment` | `object` | `string` | 3 | 0 | 0.0% | `Consumer` |
| 9 | `Country` | `object` | `string` | 1 | 0 | 0.0% | `United States` |
| 10 | `City` | `object` | `string` | 531 | 0 | 0.0% | `Henderson` |
| 11 | `State` | `object` | `string` | 49 | 0 | 0.0% | `Kentucky` |
| 12 | `Postal Code` | `int64` | `int64` | 631 | 0 | 0.0% | `42420` |
| 13 | `Region` | `object` | `string` | 4 | 0 | 0.0% | `South` |
| 14 | `Product ID` | `object` | `string` | 1,862 | 0 | 0.0% | `FUR-BO-10001798` |
| 15 | `Category` | `object` | `string` | 3 | 0 | 0.0% | `Furniture` |
| 16 | `Sub-Category` | `object` | `string` | 17 | 0 | 0.0% | `Bookcases` |
| 17 | `Product Name` | `object` | `string` | 1,850 | 0 | 0.0% | `Bush Somerset Collection Bookcase` |
| 18 | `Sales` | `float64` | `float64` | 5,825 | 0 | 0.0% | `261.96` |
| 19 | `Quantity` | `int64` | `int64` | 14 | 0 | 0.0% | `2` |
| 20 | `Discount` | `float64` | `float64` | 12 | 0 | 0.0% | `0.0` |
| 21 | `Profit` | `float64` | `float64` | 7,287 | 0 | 0.0% | `41.9136` |

---

## 3. Missingness Analysis
- Total Missing Values: 0 across all 9,994 records and 21 columns.
- Complete Cases: 9,994 (100.0%).

---

## 4. Date Coverage
- **Order Date Range:** `2014-01-03` to `2017-12-30`
  - Total Calendar Days in Span: 1,457 days
  - Unique Order Dates: 1,237 dates (trading occurred on 84.9% of calendar days)
  - Annual Transaction Count:
    - 2014: 1,993 transactions
    - 2015: 2,102 transactions
    - 2016: 2,587 transactions
    - 2017: 3,312 transactions
- **Ship Date Range:** `2014-01-07` to `2018-01-05`
  - Total Calendar Days in Span: 1,459 days
  - Unique Ship Dates: 1,334 dates
- **Shipping Lead Time (`Ship Date - Order Date`):**
  - Minimum: 0 days (Same Day delivery)
  - 25th Percentile: 3 days
  - Median (50th Percentile): 4 days
  - 75th Percentile: 5 days
  - Maximum: 7 days
  - Mean: 3.96 days
  - Violations (`Order Date > Ship Date`): 0

---

## 5. Numerical Ranges and Distributions
Verified parametric and non-parametric distribution statistics for all quantitative features:

| Metric | `Sales` ($) | `Quantity` (units) | `Discount` (rate) | `Profit` ($) |
| :--- | :--- | :--- | :--- | :--- |
| **Count** | 9,994 | 9,994 | 9,994 | 9,994 |
| **Mean** | 229.86 | 3.79 | 0.156 | 28.66 |
| **Std Dev** | 623.25 | 2.23 | 0.206 | 234.26 |
| **Minimum** | 0.444 | 1 | 0.000 | -6,599.978 |
| **25% (Q1)** | 17.280 | 2 | 0.000 | 1.729 |
| **50% (Median)** | 54.490 | 3 | 0.200 | 8.667 |
| **75% (Q3)** | 209.940 | 5 | 0.200 | 29.364 |
| **Maximum** | 22,638.480 | 14 | 0.800 | 8,399.976 |
| **IQR** | 192.660 | 3 | 0.200 | 27.635 |
| **Sum** | $2,297,200.86 | 37,873 | — | $286,397.02 |
| **Negative Count** | 0 | 0 | 0 | 1,871 (18.72%) |
| **Zero Count** | 0 | 0 | 4,798 (48.01%) | 65 (0.65%) |

---

## 6. Categorical Value Distributions

### Category (3 values)
| Category | Frequency | Percentage |
| :--- | :--- | :--- |
| Office Supplies | 6,026 | 60.30% |
| Furniture | 2,121 | 21.22% |
| Technology | 1,847 | 18.48% |

### Sub-Category (17 values)
| Sub-Category | Parent Category | Frequency | Percentage |
| :--- | :--- | :--- | :--- |
| Binders | Office Supplies | 1,523 | 15.24% |
| Paper | Office Supplies | 1,370 | 13.71% |
| Furnishings | Furniture | 957 | 9.58% |
| Phones | Technology | 889 | 8.90% |
| Storage | Office Supplies | 846 | 8.47% |
| Art | Office Supplies | 796 | 7.96% |
| Accessories | Technology | 775 | 7.75% |
| Chairs | Furniture | 617 | 6.17% |
| Appliances | Office Supplies | 466 | 4.66% |
| Labels | Office Supplies | 364 | 3.64% |
| Tables | Furniture | 319 | 3.19% |
| Envelopes | Office Supplies | 254 | 2.54% |
| Bookcases | Furniture | 228 | 2.28% |
| Fasteners | Office Supplies | 217 | 2.17% |
| Supplies | Office Supplies | 190 | 1.90% |
| Machines | Technology | 115 | 1.15% |
| Copiers | Technology | 68 | 0.68% |

### Segment (3 values)
| Segment | Frequency | Percentage |
| :--- | :--- | :--- |
| Consumer | 5,191 | 51.94% |
| Corporate | 3,020 | 30.22% |
| Home Office | 1,783 | 17.84% |

### Region (4 values)
| Region | Frequency | Percentage |
| :--- | :--- | :--- |
| West | 3,203 | 32.05% |
| East | 2,848 | 28.50% |
| Central | 2,323 | 23.24% |
| South | 1,620 | 16.21% |

### Ship Mode (4 values)
| Ship Mode | Frequency | Percentage | Typical Lead Time |
| :--- | :--- | :--- | :--- |
| Standard Class | 5,968 | 59.72% | 3 to 7 days |
| Second Class | 1,945 | 19.46% | 1 to 5 days |
| First Class | 1,538 | 15.39% | 1 to 4 days |
| Same Day | 543 | 5.43% | 0 to 1 days |

### Top 5 States (out of 49)
1. California: 2,001 records (20.02%)
2. New York: 1,128 records (11.29%)
3. Texas: 985 records (9.86%)
4. Pennsylvania: 587 records (5.87%)
5. Washington: 506 records (5.06%)

### Top 5 Cities (out of 531)
1. New York City: 915 records (9.16%)
2. Los Angeles: 747 records (7.47%)
3. Philadelphia: 537 records (5.37%)
4. San Francisco: 510 records (5.10%)
5. Seattle: 428 records (4.28%)
