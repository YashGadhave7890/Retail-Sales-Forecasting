# Data Cleaning Report

## Input Dataset
- **Raw File:** `data/raw/Sample - Superstore.csv`
- **File Size:** 2,298,597 bytes (~2.19 MB)
- **Verified SHA-256 Hash:** `c24c0548e7cf0f20b7a35d22941bb7d751c910d0c39bbfcfae850331ed1c15ce`
- **Immutability Status:** 100% Intact and Unmodified

---

## Cleaning Objectives
The objective of Phase 5 is to establish a transparent, reproducible, and business-aware data-cleaning pipeline that:
1. Validates and resolves confirmed data-quality issues without aggressive, unjustified row deletion.
2. Distinguishes genuine business outcomes (e.g., negative profit from steep discounts) from system errors.
3. Investigates duplicate identifiers to separate accidental database double-entries from legitimate multi-line order fulfillments.
4. Standardizes text encodings, trims typographical whitespace, and formats geospatial and temporal data types.
5. Produces an auditable, clean processed dataset at `data/processed/superstore_cleaned.csv` ready for subsequent EDA, feature engineering, and time-series forecasting.

---

## Data Quality Issues Investigated
Based on Phase 4 data validation findings, four key potential data-quality areas were investigated:
1. **Surrogate Key vs. Business Duplicate Inconsistencies:** Why 8 `(Order ID, Product ID)` pairs (16 rows) appeared in the dataset.
2. **Typographical Whitespace:** 16 records in `Product Name` containing trailing spaces.
3. **Negative Profits:** 1,871 transactions with negative profits.
4. **Geospatial & Temporal Data Types:** 4-digit numeric postal codes and unparsed date objects.

---

## Duplicate Analysis

### Empirical Investigation of Duplicate Order-Product Combinations
The raw dataset contained 8 pairs of rows (16 total rows) sharing the same `(Order ID, Product ID)`:
1. `CA-2015-103135` (`OFF-BI-10000069`): Row 6499 (Qty: 9, Sales: $135.09) vs. Row 6501 (Qty: 6, Sales: $90.06).
2. `CA-2016-129714` (`OFF-PA-10001970`): Row 351 (Qty: 2, Sales: $24.56) vs. Row 353 (Qty: 4, Sales: $49.12).
3. `CA-2016-137043` (`FUR-FU-10003664`): Row 1301 (Qty: 6, Sales: $572.76) vs. Row 1302 (Qty: 3, Sales: $286.38).
4. `CA-2016-140571` (`OFF-PA-10001954`): Row 9169 (Qty: 14, Sales: $319.76) vs. Row 9170 (Qty: 2, Sales: $45.68).
5. `CA-2017-118017` (`TEC-AC-10002006`): Row 7882 (Qty: 6, Sales: $76.75) vs. Row 7883 (Qty: 8, Sales: $102.34).
6. `CA-2017-152912` (`OFF-ST-10003208`): Row 3184 (Qty: 9, Sales: $1633.14) vs. Row 3185 (Qty: 3, Sales: $544.38).
7. `US-2016-123750` (`TEC-AC-10004659`): Row 431 (Qty: 7, Sales: $408.74) vs. Row 432 (Qty: 5, Sales: $291.96).
8. `US-2014-150119` (`FUR-CH-10002965`): Row 3406 (Qty: 2, Sales: $281.372, Discount: 0.3, Profit: -$12.0588) vs. Row 3407 (Qty: 2, Sales: $281.372, Discount: 0.3, Profit: -$12.0588).

### Decision & Evidence
- In **Pairs 1 through 7 (14 records)**, the transaction quantities and sales values are completely different. These represent legitimate multi-package shipments or split purchase lines within a single customer order. They were **intentionally retained**.
- In **Pair 8 (Rows 3406 and 3407)**, all 20 business attributes—Customer ID (`LB-16795`), Product (`FUR-CH-10002965`), Quantity (`2`), Sales (`$281.372`), Discount (`0.3`), and Profit (`-$12.0588`)—are 100% identical. The difference was merely an auto-incremented `Row ID` surrogate key. This is confirmed as an accidental double-entry logging duplicate.
- **Action Taken:** Removed Row ID 3407 (second occurrence) and preserved Row ID 3406.

---

## Text Standardization
- Applied `.str.strip()` across all categorical and text columns (`Product Name`, `Category`, `Sub-Category`, `Customer Name`, `Segment`, `City`, `State`, `Region`, `Ship Mode`).
- Successfully eliminated trailing whitespace from all 16 affected `Product Name` entries.
- Verified that all 531 unique city names have 1-to-1 matching casing with their lowercase equivalents (`df['City'].nunique() == df['City'].str.lower().nunique() == 531`). Zero casing anomalies detected.

---

## Numerical Validation
- **Sales:** Verified all values $> 0$ (range: $\$0.444$ to $\$22,638.48$). Extreme values represent high-ticket enterprise technology/copier purchases and were not clipped.
- **Quantity:** Verified all values are positive integers $\ge 1$ (range: 1 to 14 units).
- **Discount:** Verified bounded strictly within $[0.0, 0.8]$.
- **Profit:** 
  - Verified range: $-\$6,599.98$ to $\$8,399.98$.
  - Analysis confirmed that negative-profit records carry an average discount of $37.8\%$ (compared to $10.5\%$ on profitable transactions).
  - Negative profits reflect genuine commercial realities (loss-leader promotions, margin destruction from deep discounts).
  - **Action Taken:** All 1,870 remaining negative-profit transactions were **strictly retained**.

---

## Date Validation
- Standardized `Order Date` and `Ship Date` into `datetime64[ns]`.
- Verified that 100% of records satisfy the invariant `Order Date <= Ship Date`.
- Shipping lead time ranges strictly from 0 days (Same Day) to 7 days (Standard Class), with a mean of 3.96 days. Zero artificial dates were created.

---

## Records Removed
- **Total Records Removed:** 1
- **Specific Record Removed:** `Row ID 3407` (`Order ID: US-2014-150119`, Customer `LB-16795`)
- **Reason:** Confirmed exact duplicate double-entry of `Row ID 3406` across all 20 business attributes.
- **Rule Applied:** `df.duplicated(subset=[col for col in df.columns if col != 'Row ID'], keep='first')`.

---

## Records Retained
- **Total Records Retained:** 9,993 out of 9,994 ($99.99\%$).
- **Multi-Line Order Records:** All 14 split order-item rows preserved.
- **Negative Profit Transactions:** All 1,870 genuine loss transactions preserved.
- **Extreme High-Value Orders:** All genuine high-value sales preserved.

---

## Transformations Applied
1. **Deduplication:** Dropped accidental duplicate row `Row ID 3407`.
2. **Text Normalization:** Trimmed leading/trailing whitespace across all text columns.
3. **Geospatial Standardization:** Converted `Postal Code` from float/integer into standardized 5-digit zero-padded strings (`zfill(5)`), preserving leading zeros for East Coast postal codes (e.g. `01040`).
4. **Temporal Standardization:** Parsed `Order Date` and `Ship Date` to `datetime64[ns]`.
5. **Output Serialization:** Exported to `data/processed/superstore_cleaned.csv` with UTF-8 encoding and clean index suppression.

---

## Before vs After Summary

| Metric | Before Cleaning (Raw) | After Cleaning (Processed) | Delta / Meaning |
| :--- | :--- | :--- | :--- |
| **Row Count** | 9,994 | 9,993 | -1 (Removed accidental clone 3407) |
| **Column Count** | 21 | 21 | 0 (Exact schema preserved) |
| **Missing Values** | 0 | 0 | 0 (Completely clean) |
| **Exact Business Duplicates** | 1 pair (2 rows) | 0 | -1 pair (Deduplicated) |
| **Split Order Items** | 7 pairs (14 rows) | 7 pairs (14 rows) | 0 (Fully preserved) |
| **Product Name Whitespace Issues**| 16 rows | 0 rows | -16 (Normalized) |
| **Postal Code Format** | Integer (`int64`) | 5-digit string (`object`) | Standardized geospatial format |
| **Date Validity** | 100% valid | 100% valid | Preserved |
| **Total Revenue (Sales)** | $2,297,200.86 | $2,296,919.49 | -$281.37 (Removed duplicate row) |
| **Total Profit** | $286,397.02 | $286,409.08 | +$12.06 (Removed duplicate loss) |
| **Negative Profit Records** | 1,871 rows | 1,870 rows | -1 row (Removed duplicate loss) |

---

## Assumptions
1. **Line-Item Fulfillment Assumption:** Multi-item orders sharing `(Order ID, Product ID)` with distinct quantities represent legitimate distinct packaging or split allocations rather than error.
2. **Loss-Leader Pricing Assumption:** Transactions with steep discounts yielding negative net profits reflect real business loss-leader tactics rather than recording flaws.

---

## Limitations
1. **Unobserved Cancellations/Returns:** The dataset does not include a separate order-status or returns sheet in this single-table export; all listed sales represent settled transactions.
2. **Daily Resampling Gaps:** Order transactions occur on 1,237 distinct dates across 1,457 calendar days. For subsequent daily time-series forecasting, regular calendar frequency expansion will be addressed in Phase 7 (Feature Engineering).

---

## Reproducibility
The entire cleaning procedure is automated and reproducible. From the project root, execute:
```bash
.venv\Scripts\python.exe -m src.data_cleaning
```
Output is deterministically saved to `data/processed/superstore_cleaned.csv`.
