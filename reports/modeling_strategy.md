# Modeling Strategy & Chronological Validation Design

**Project:** Retail Sales Forecasting & Analytics  
**Phase:** Phase 8 – Baseline & Machine Learning Forecasting Models  
**Status:** Completed & Validated  

---

## Business Objective
The primary business objective is to deliver accurate, reliable, and interpretable weekly retail sales forecasts. The models assist retail managers in:
1. **Replenishment & Procurement:** Determining warehouse inventory purchasing ahead of seasonal peaks (specifically the Q4 holiday surge).
2. **Logistical Capacity:** Allocating labor, shipping resources, and carrier capacity across fulfillment centers.
3. **Cash-Flow Planning:** Providing predictable quarterly revenue baselines for executive working capital management.

---

## Chronological Train / Test Partitioning

Standard random cross-validation (e.g. K-Fold) is strictly prohibited in time-series forecasting because randomly shuffling records allows future observations to leak into the past, artificially deflating error metrics and causing disastrous real-world performance.

To guarantee zero look-ahead bias, all evaluations adhere to a **strict chronological partition**:

```
[--------------------- 104 WEEKS TRAINING (2015 - 2016) ---------------------] [----- 53 WEEKS HOLDOUT TEST (2017) -----]
2015-01-04                                                          2016-12-25  2017-01-01                       2017-12-31
```

| Partition | Start Date | End Date | Observation Count | Calendar Span | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Development / Training Set** | `2015-01-04` | `2016-12-25` | **104 weeks** (66.2%) | 2 full calendar years | Model fitting and hyperparameter cross-validation |
| **Final Out-of-Time Test Set** | `2017-01-01` | `2017-12-31` | **53 weeks** (33.8%) | 1 full calendar year | Unbiased, held-out evaluation of generalized performance |
| **Total Usable Series** | `2015-01-04` | `2017-12-31` | **157 weeks** | 3 complete calendar years | Complete series with 52-week lag history |

---

## Expanding-Window Cross-Validation Design

For model development and comparison within the training window (2015–2016), we employ an expanding-window **TimeSeriesSplit** with 5 folds:
- Fold 1: Train on weeks 1–19 (Jan 2015 – May 2015), Validate on weeks 20–36 (May 2015 – Aug 2015) [17 weeks]
- Fold 2: Train on weeks 1–36 (Jan 2015 – Aug 2015), Validate on weeks 37–53 (Sep 2015 – Dec 2015, capturing Q4 2015) [17 weeks]
- Fold 3: Train on weeks 1–53 (Jan 2015 – Dec 2015), Validate on weeks 54–70 (Jan 2016 – Apr 2016) [17 weeks]
- Fold 4: Train on weeks 1–70 (Jan 2015 – Apr 2016), Validate on weeks 71–87 (May 2016 – Aug 2016) [17 weeks]
- Fold 5: Train on weeks 1–87 (Jan 2015 – Aug 2016), Validate on weeks 88–104 (Sep 2016 – Dec 2016, capturing Q4 2016) [17 weeks]

In every fold:
1. Training data contains strictly earlier dates than validation data.
2. The validation set is never seen during model fitting.
3. Feature scalers and transformers are fitted **strictly** on the training slice.

---

## Prediction Horizon & Operational Protocol

- **Operational Planning Horizon ($H$):** 1 to 12 weeks ahead.
- **Short-Term Horizon (1 to 4 weeks):** Used for weekly inventory adjustments and staffing.
- **Medium-Term Horizon (5 to 12 weeks):** Used for quarterly purchasing and supplier lead times.

---

## One-Step vs Multi-Step Limitation Documentation

> [!IMPORTANT]
> **Explicit Architectural Limitation:**
> In this phase, the supervised feature dataset evaluates **one-step-ahead rolling walk-forward forecasting** ($h=1$). At each week $t$ in the test set, the models generate a forecast using predictor features constructed from data observed up to week $t-1$.
>
> While this satisfies the primary retail demand-tracking workflow, generating a true multi-step direct forecast (e.g. predicting week $t+12$ without observing weeks $t+1 \dots t+11$) requires either:
> 1. **Recursive Forecasting:** Feeding the model's own predictions back as future lag inputs.
> 2. **Direct Multi-Output Modeling:** Training distinct models for each horizon $h \in \{1, 2, \dots, 12\}$.
> 
> We explicitly document this operational boundary rather than claiming the one-step supervised model represents a native 12-step direct forecast.

---

## Evaluation Metrics

To evaluate forecast accuracy from both absolute scale and percentage perspective, four standard time-series metrics are tracked:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|$$
   Measures average forecast error in dollar terms ($). Robust to occasional extreme surges.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$
   Penalizes large errors disproportionately, indicating model stability during peak demand weeks.

3. **Weighted Absolute Percentage Error (WAPE):**
   $$\text{WAPE} = \frac{\sum_{t=1}^N |y_t - \hat{y}_t|}{\sum_{t=1}^N y_t} \times 100$$
   Industry-standard retail percentage metric. Avoids division-by-zero or distortion from low-volume sales weeks.

4. **Mean Absolute Percentage Error (MAPE):**
   $$\text{MAPE} = \frac{1}{N} \sum_{t=1}^N \left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100$$
   Standard percentage error calculated across non-zero actuals.
