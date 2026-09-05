# Raw Data Validation Report

## File Information
- **File Path:** `C:/Users/yashg/OneDrive/Desktop/Retail-Sales-Forecasting/data/raw/Sample - Superstore.csv`
- **File Size:** 2,298,597 bytes (2.19 MB)
- **Character Encoding:** `windows-1252`
- **Cryptographic Hash (SHA-256):** `c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce`

## Dataset Dimensions
- **Row Count:** 9,994
- **Column Count:** 21
- **Status:** PASS (Matches expected 9,994 rows x 21 columns)

## Schema
All 21 columns detected:
| Column Name | Expected | Detected | Data Type in CSV |
| :--- | :--- | :--- | :--- |
| `Row ID` | Yes | Yes | Present |
| `Order ID` | Yes | Yes | Present |
| `Order Date` | Yes | Yes | Present |
| `Ship Date` | Yes | Yes | Present |
| `Ship Mode` | Yes | Yes | Present |
| `Customer ID` | Yes | Yes | Present |
| `Customer Name` | Yes | Yes | Present |
| `Segment` | Yes | Yes | Present |
| `Country` | Yes | Yes | Present |
| `City` | Yes | Yes | Present |
| `State` | Yes | Yes | Present |
| `Postal Code` | Yes | Yes | Present |
| `Region` | Yes | Yes | Present |
| `Product ID` | Yes | Yes | Present |
| `Category` | Yes | Yes | Present |
| `Sub-Category` | Yes | Yes | Present |
| `Product Name` | Yes | Yes | Present |
| `Sales` | Yes | Yes | Present |
| `Quantity` | Yes | Yes | Present |
| `Discount` | Yes | Yes | Present |
| `Profit` | Yes | Yes | Present |

- **Missing Columns:** None
- **Unexpected Columns:** None
- **Schema Check:** PASS

## Date Coverage
- **Order Date Range:** `2014-01-03` to `2017-12-30`
- **Ship Date Range:** `2014-01-07` to `2018-01-05`
- **Unique Order Dates:** 1237 distinct trading dates across 1,457 calendar days
- **Date Chronology Check (`Order Date <= Ship Date`):** PASS
- **Shipping Duration:** Min = 0 days, Max = 7 days, Mean = 3.96 days
- **Date Validation:** PASS

## Missing Values
- **Total Missing Values:** 0
- **Columns with Nulls:** None (0 nulls across all 21 columns)
- **Missing Values Check:** PASS

## Duplicate Analysis
- **Full Row Duplicates:** 0
- **Unique Orders:** 5009
- **Multi-Item Order Records:** 4985 rows belong to multi-line orders (expected in retail transactional data)
- **Duplicate Order ID + Product ID Combinations:** 8 occurrences (16 rows) where the exact same Product ID appears more than once in the same Order ID (e.g. separate line items or reorders)
- **Duplicate Rows Check:** PASS

## Numerical Validation
| Metric | Range / Distribution | Validation Rule | Status |
| :--- | :--- | :--- | :--- |
| **Sales** | Min: $0.444, Max: $22,638.48, Mean: $229.86 | `Sales > 0` | PASS |
| **Quantity** | Min: 1, Max: 14 units | `Quantity >= 1` | PASS |
| **Discount** | Min: 0.0, Max: 0.8 | `0.0 <= Discount <= 1.0` | PASS |
| **Profit** | Min: $-6599.98, Max: $8399.98, Mean: $28.66 | Domain Check | Verified (1871 negative rows / 18.72%) |

- **Numerical Sanity Check:** PASS

## Date Validation
- Order dates span 4 complete calendar years (2014, 2015, 2016, 2017).
- Ship dates extend into early January 2018 for late December 2017 orders, strictly obeying the positive shipping lead-time invariant.
- No invalid, unparseable, or out-of-order date entries were found.

## Categorical Validation
- **Categories (3):** Furniture, Office Supplies, Technology (PASS)
- **Sub-Categories (17):** 17 distinct sub-categories (PASS)
- **Segments (3):** Consumer, Corporate, Home Office (PASS)
- **Regions (4):** Central, East, South, West (PASS)
- **Ship Modes (4):** First Class, Same Day, Second Class, Standard Class (PASS)
- **Geographic Coverage:** 49 states, 531 cities (PASS)
- **Customer Coverage:** 793 unique customers (PASS)
- **Product Catalog:** 1862 unique product IDs (PASS)
- **Whitespace Check:** Found trailing/leading whitespace in `Product Name` (16 rows). All other categorical columns are completely clean.

## Data Integrity Findings

### Checks That Passed
1. **Raw File Immutability & Hash:** Raw CSV verified intact with SHA-256 hash `c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce`.
2. **Schema & Dimensions:** Exactly 9,994 rows and 21 columns detected matching the target schema.
3. **Completeness:** 0 missing values across all 21 columns.
4. **No Row Duplicates:** Zero identical duplicate rows.
5. **Temporal Consistency:** 100% of rows have `Order Date <= Ship Date`. All shipping durations range from 0 to 7 days.
6. **Valid Sales & Quantities:** No zero or negative sales; no zero or negative quantities.
7. **Valid Discounts:** All discount values fall strictly within [0.0, 0.8].

### Suspicious Records Requiring Future Investigation / Cleaning
1. **Duplicate Product Lines within Same Order:** Exactly 8 Order ID + Product ID pairs (16 rows) appear twice in the same order. During Phase 5 (Data Cleaning), we will determine whether these represent separate line-item deliveries, split shipments, or true data logging duplicates.
2. **Whitespace in Product Names:** 16 records have whitespace irregularities in `Product Name` (e.g. trailing space), which should be trimmed during preprocessing.
3. **Negative Profit Rows:** 1,871 transactions (18.72%) generate negative profit. These are genuine business losses tied to steep discounts (e.g. >20% discounts), not data corruption.
4. **Irregular Daily Order Cadence:** While 4 full years are represented, trading occurs on 1,237 distinct dates out of 1,457 calendar days. For daily time-series forecasting, regular calendar resampling with zero-demand imputation will be required.

## Conclusion
The raw Superstore dataset passed all fundamental integrity and schema verification checks. The dataset is fully reproducible, genuine, structurally sound, and ready for clean ingestion. No data cleaning, feature engineering, modeling, or transformations have been performed in accordance with raw data immutability principles.
