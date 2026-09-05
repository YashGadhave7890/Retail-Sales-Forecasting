# Final Forecasting Model Selection Report

## 1. Decision Summary & Selected Candidate Model

**Selected Final Model:** **Ridge Regression** (Standardized L2-Regularized Linear Regression, $\alpha=10.0$)

After comprehensive chronological cross-validation (5 expanding folds), out-of-sample holdout testing (53 weeks of 2017), residual diagnostics, and robustness backtesting, **Ridge Regression** is selected as the primary forecasting engine for the Retail Sales Forecasting system.

> [!NOTE]
> **Deliberate Selection Rationale:**
> Ridge Regression is not selected merely because it achieved the lowest point error on the 2017 holdout test set. Rather, it is selected because across multiple independent dimensions—cross-validation error, backtesting stability, sample-size parsimony, directional interpretability, and computational footprint—it demonstrated the most consistent and defensible behavior for this small weekly retail dataset.

---

## 2. Multi-Dimensional Candidate Model Evaluation

The table below summarizes the empirical metrics across all 6 evaluated models:

| Evaluation Dimension | Naive ($t-1$) | Seasonal Naive ($t-52$) | Ridge Regression | Random Forest | HistGradientBoosting | Gradient Boosting |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CV Mean MAE ($)** | $5,366.34 | $5,342.45 | **$5,130.83** | $5,196.86 | $5,151.23 | $5,733.63 |
| **CV Mean RMSE ($)** | $6,751.16 | $6,864.83 | **$6,554.35** | $6,602.18 | $6,778.72 | $7,370.19 |
| **CV Mean WAPE (%)** | 49.61% | 49.14% | 48.60% | 47.12% | **46.52%** | 53.07% |
| **Holdout 2017 MAE ($)** | $7,187.28 | $7,075.47 | **$5,377.97** | $5,600.20 | $5,926.80 | $5,493.08 |
| **Holdout 2017 RMSE ($)**| $9,134.80 | $9,172.35 | **$6,778.95** | $7,289.91 | $7,601.07 | $7,125.67 |
| **Holdout 2017 WAPE (%)**| 51.56% | 50.76% | **38.58%** | 40.18% | 42.52% | 39.41% |
| **Fold Stability (Std MAE)**| $1,701.11 | $1,708.61 | **$1,684.12** | $1,862.72 | $1,766.76 | **$1,676.37** |
| **Interpretability** | Trivial | Trivial | **High (Linear Coeffs)** | Moderate | Low | Moderate |
| **Training Time (s)** | 0.000s | 0.000s | **0.003s** | 0.084s | 0.088s | 0.055s |
| **Serialized Size** | N/A | N/A | **2.5 KB** | 316 KB | 131 KB | 118 KB |

---

## 3. Detailed Justification for Selection

### 3.1 Consistent Generalization Across Both CV and Holdout
- During model development across 5 expanding folds (2015–2016), Ridge achieved the lowest Mean MAE ($5,130.83) and lowest Mean RMSE ($6,554.35).
- On the unseen 2017 test set, Ridge maintained this lead, generating the lowest Test MAE ($5,377.97), lowest Test RMSE ($6,778.95), and lowest Test WAPE (38.58%).
- It beat the naive persistence baseline by **25.17%** and the seasonal naive baseline by **23.99%**.

### 3.2 Sample-Size Parsimony & Overfitting Protection
- The entire usable forecasting history consists of **157 weekly observations** (104 training, 53 testing).
- Deep tree ensembles (Random Forest, Gradient Boosting) risk partitioning 104 samples into noisy, overfit leaf regimes, particularly during extreme holiday periods.
- Ridge Regression constrains model complexity through an L2 penalty ($\alpha=10.0$), shrinking redundant multi-lag collinearities smoothly across correlated features (`sales_lag_1` through `sales_lag_12`).

### 3.3 Directional Business Interpretability
- Unlike tree ensembles that only offer non-directional feature importances, Ridge provides direct linear coefficients:
  - Positive coefficients clearly denote revenue-driving momentum (`orders_lag_1`, `sales_lag_52`).
  - Negative coefficients clearly denote drag factors (e.g., deep promotional discount saturation).
- This transparency is vital for executive trust and pairs naturally with business domain knowledge.

### 3.4 Diagnostic Soundness
- Rigorous hypothesis testing on 2017 residuals confirmed that Ridge has **no statistically significant global bias** ($t = -1.4339, p = 0.1576$), exhibits **normal residuals** ($W = 0.9751, p = 0.3302$), and leaves **virtually zero autocorrelation** in the error series ($d = 2.0597$).

---

## 4. Known Limitations & Competing Model Discussion

### 4.1 Competing Models Not Selected
1. **HistGradientBoosting:** Highly competitive on CV WAPE (46.52%), but exhibited higher error on extreme 2017 holiday spikes (Test MAE: $5,926.80), with less transparency.
2. **Random Forest:** Solid median performance ($4,061.90), but exhibited higher cross-fold variance (Std MAE: $1,862.72) and larger artifact size.
3. **Gradient Boosting:** Strongest tree alternative on holdout test ($5,493.08 MAE), but showed higher cross-validation error ($5,733.63 MAE).

### 4.2 Residual Limitations of the Selected Model
- **Peak Underprediction:** Ridge underpredicts extreme holiday spikes (>75th percentile sales) by an average of +$5,810 per week because linear hyperplanes cannot bend sharply to capture non-linear demand explosions.
- **Trough Overprediction:** Post-holiday slump periods are overpredicted by -$7,079 per week due to autoregressive lag inertia.
- **Point Forecast Nature:** The model outputs point forecasts. Inventory planners must combine these predictions with empirical error distribution quantiles for risk mitigation.
