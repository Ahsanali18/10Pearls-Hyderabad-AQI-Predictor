# Hyderabad AQI Prediction System — Project Report

**Project:** Multi-Horizon Air Quality Index Forecasting System
**Location:** Hyderabad, Sindh, Pakistan (25.396°N, 68.358°E)
**Organization:** 10Pearls — Internship Project
**Forecast Horizons:** Day 1 (next 24h mean), Day 2 (next 48h mean), Day 3 (next 72h mean)
**Data Period:** January 4, 2025 → May 27, 2026 (16 months · 12,352 hourly rows)
**Live Dashboard:** 🔗 [View Live Application](https://10pearls-hyderabad-aqi-predictor-xgct4ppjp9s5qmyqucwxzv.streamlit.app/)

## 1. Background

Hyderabad, the second-largest city in Sindh, Pakistan, regularly experiences moderate to unhealthy air quality driven by vehicular traffic, industrial activity, dust storms, and seasonal temperature inversions. Residents, healthcare workers, and urban planners lack access to localized multi-day AQI forecasts — existing services either focus on Karachi or provide only current readings without future context.

This project builds a fully automated, production-grade AQI forecasting system that predicts the **mean AQI for the next three calendar days** (Day 1, Day 2, Day 3) using a separate machine learning model per horizon. The system runs entirely on free-tier infrastructure — GitHub Actions, MongoDB Atlas, Streamlit Cloud — and requires zero manual intervention after deployment.

---

## 2. Objective

Build a fully automated, end-to-end pipeline that:

1. Collects hourly weather and air quality observations from Open-Meteo APIs
2. Engineers 60 physically meaningful features grounded in EDA findings
3. Trains three separate models (LinearRegression, RandomForest, XGBoost) — one dedicated model per forecast horizon
4. Applies horizon-aware feature filtering to remove irrelevant short-lag features for longer horizons
5. Generates non-overlapping 24h mean AQI forecasts for Day 1, Day 2, and Day 3
6. Serves a public-facing live dashboard updated every hour
7. Runs continuously through scheduled CI/CD workflows without any human intervention

---

## 3. Data Source

All data comes from **Open-Meteo**, a free, open-source weather and air quality API:

- **Hourly weather:** temperature, humidity, wind speed, wind direction, precipitation, surface pressure, weather code
- **Hourly air quality:** PM2.5, PM10, NO₂, SO₂, ozone, CO, US AQI

Air quality data is derived from the **Copernicus Atmosphere Monitoring Service (CAMS)**, which runs global atmospheric composition models at ~11 km resolution and updates once daily. Weather data updates every 1–6 hours.

**Coordinates:** 25.396°N, 68.358°E — Hyderabad, Sindh, Pakistan

---

## 4. Approach

### 4.1 Feature Engineering

The model uses 60 features grouped into 11 categories. Every feature was justified through the exploratory data analysis (EDA) before being added to the pipeline:

| Group | Count | Features | EDA Justification |
|---|---|---|---|
| Weather current | 8 | temperature, humidity, rain, wind_speed, pressure, weather_code, wind_dir_sin, wind_dir_cos | wind_speed (r=−0.369), temperature (r=−0.386). Cold+calm = worst AQI. Wind direction encoded as sin/cos — circular encoding handles 0°=360° wrap correctly |
| Pollutants current | 6 | pm2_5, pm10, no2, o3, so2, co | pm2_5 (r=+0.720) — strongest individual predictor. All pollutants retained as they measure distinct emission sources |
| Future weather t+24 | 5 | temp_24h, wind_24h, pressure_24h, humidity_24h, pm2_5_24h | Day 1 forecast context. Fetched from Open-Meteo 4-day forecast at inference. Learned from shifted historical values during training |
| Future weather t+36 | 5 | temp_36h, wind_36h, pressure_36h, humidity_36h, pm2_5_36h | Bridge features — midpoint between Day 1 and Day 2 horizons. Added specifically after discovering Day 2 accuracy was weaker without a midpoint weather anchor |
| Future weather t+48 | 5 | temp_48h, wind_48h, pressure_48h, humidity_48h, pm2_5_48h | Day 2 forecast context |
| Future weather t+72 | 5 | temp_72h, wind_72h, pressure_72h, humidity_72h, pm2_5_72h | Day 3 forecast context |
| Cyclical time | 6 | hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos | Sin/cos preserves circular nature: hour 23 and hour 0 are 1h apart, not 23h. EDA confirmed clear diurnal AQI pattern (peak 19:00, trough 10:00) and strong monsoon seasonality |
| Time flags | 5 | is_night, is_morning_rush, is_afternoon, is_evening_rush, is_weekend | Binary flags allow tree models to learn sharp period boundaries. EDA showed peak AQI at 19:00 (AQI=95.1) and lowest at 10:00 (AQI=84.5) |
| Lag features | 8 | aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h, aqi_lag_36h, aqi_lag_48h, aqi_lag_72h | EDA autocorrelation: r=0.992 (1h), r=0.743 (24h), r=0.508 (72h). aqi_lag_36h added as Day 2 bridge anchor after EDA showed midpoint correlation mattered |
| Rolling statistics | 3 | aqi_rolling_mean_24h, aqi_rolling_std_24h, aqi_rolling_mean_6h | Mean captures pollution persistence; std flags volatile episodes where AQI is changing rapidly |
| Derived / interaction | 4 | aqi_change_rate, dispersion_index, heat_humidity, pressure_change | dispersion_index = wind_speed × temperature (physics-based atmospheric dispersal proxy). pressure_change detects approaching fronts. heat_humidity captures humid-heat combined effect |

**Excluded features:**

- `us_aqi` — target/output variable
- `wind_direction` (raw degrees) — consumed to produce `wind_dir_sin` / `wind_dir_cos`, then dropped
- Lags beyond 72h — r < 0.5, adds noise beyond the forecasting horizon

### 4.2 Target Definition

Targets are **non-overlapping daily mean AQI windows**, computed at each timestamp t:

```
aqi_24h = mean(AQI[t+1  : t+24])   ← Day 1 — next 24h average
aqi_48h = mean(AQI[t+25 : t+48])   ← Day 2 — 25th to 48th hour average
aqi_72h = mean(AQI[t+49 : t+72])   ← Day 3 — 49th to 72nd hour average
```

**Why non-overlapping:** Overlapping targets (e.g., target_48h covering t+1 to t+48) share rows with target_24h. A model trained on overlapping targets would indirectly learn from shared future data — inflating apparent performance and producing correlated predictions. Non-overlapping windows ensure each target represents genuinely distinct future information.

**Why mean AQI, not hourly:** Predicting a 24-hour mean reduces the impact of transient hour-to-hour fluctuations, produces smoother and more actionable forecasts for residents (people plan their day, not individual hours), and makes the model's task more tractable at longer horizons.

### 4.3 Horizon-Aware Feature Filtering

A key insight: **short-lag features carry no predictive signal for longer horizons.** The AQI reading from 1 hour ago is useful for predicting tomorrow's AQI, but entirely useless for predicting the average AQI 49–72 hours from now.

| Horizon | Features Used | Removed |
|---|---|---|
| 24h | 60 | None |
| 48h | 57 | aqi_lag_1h, aqi_lag_3h, aqi_change_rate |
| 72h | 55 | Above + aqi_lag_6h, aqi_lag_12h |

Each model is trained and evaluated on only the features relevant to its horizon. The live pipeline applies the same filtering at inference time — the feature set at prediction time always matches what the model was trained on.

### 4.4 Training vs Inference Feature Pipeline

The system uses two separate feature computation paths — a critical design decision to prevent data leakage while enabling real-time prediction.

**Training (Historical data):**
- Future weather columns computed via `shift(-N)` on historical data
  - e.g., `temp_48h = temperature.shift(-48)` — actual temperature 48 hours later
- Targets computed using the non-overlapping window formula
- Model learns the true relationship between future weather and future AQI

**Inference (live hourly):**
- Future weather columns filled from Open-Meteo forecast API
  - `temp_48h` = Open-Meteo forecast temperature for t+48h
- Lag features computed from last 72 rows stored in MongoDB
- Rolling statistics computed from the same MongoDB history

The model learns on ground truth (actual future temperatures) and predicts using forecasts (best available approximation of future temperatures).

### 4.5 Model Selection

Three models were trained per horizon — 9 total:

| Model | Scaling Required | Notes |
|---|---|---|
| LinearRegression | Yes (StandardScaler) | Linear baseline — important for detecting when relationships are not complex |
| RandomForest | No | Tree ensemble, low overfitting risk |
| XGBoost | No | Gradient boosting — typically strongest on tabular data |

The progression from 24h to 72h uses shallower trees (max_depth 5→3), fewer estimators (500→200), and stronger regularization (reg_lambda 1.0→5.0). This directly counteracts the overfitting risk that grows as the target becomes harder to predict.

**Each horizon gets its own StandardScaler** fitted on that horizon's training split and saved in the model bundle — ensuring the live pipeline applies identical scaling at inference.

### 4.6 Model Registry

After training, all three models are saved to MongoDB via GridFS along with their metadata:

```python
bundle = {
    "model":           model,           # fitted model object
    "scaler":          scaler,          # StandardScaler for this horizon
    "model_name":      model_name,      # "XGBoost" / "RandomForest" / "LinearRegression"
    "target":          target,          # "aqi_24h" / "aqi_48h" / "aqi_72h"
    "feature_columns": feat_cols,       # filtered list — 60 / 57 / 55
    "is_best":         is_best,         # True for lowest RMSE per horizon
}
```

---

## 5. Exploratory Data Analysis

A comprehensive Exploratory Data Analysis — EDA was conducted to ground every feature engineering decision in data evidence:

1. **AQI Distribution** — Predominantly Moderate (71.2%), with USG episodes (21.2%). Mean AQI ≈ 87, median ≈ 80 — right-skewed due to pollution spikes. Only 3.6% of hours have Good air quality.

2. **Strong Temporal Persistence** — r=0.992 at 1h lag, r=0.743 at 24h lag, r=0.508 at 72h lag. AQI is highly autocorrelated, justifying lag features as the backbone of the feature set.

3. **Weather Correlations** — wind_speed (r=−0.369), temperature (r=−0.386) — strongest weather predictors. pressure (r=+0.450) — high pressure traps pollutants under inversions.

4. **PM2.5 Dominance** — r=+0.720 — strongest single-feature correlation with AQI. Confirmed role as primary pollutant feature.

5. **Clear Diurnal Cycle** — Peak AQI at 19:00 (95.1), lowest at 10:00 (84.5). Daily variation of 10.6 AQI units — directly justified the time-of-day flags.

6. **Strong Seasonal Variation** — Winter mean=106.3 vs Monsoon mean=74.3 — a 32-point gap. 

7. **No Weekend Effect** — Weekday 86.4 vs Weekend 87.6 — only 1.2 point difference. AQI in Hyderabad is driven by regional atmospheric conditions, not local traffic patterns. 

8. **Outlier Analysis** — Outliers identified via IQR method were retained as real atmospheric events (dust storms, temperature inversions) — not data errors.

---

## 6. Results

### All Models — Full Comparison

| Horizon | Model | Test MAE | Test RMSE | Test R² | Status |
|---|---|---|---|---|---|
| 24h | XGBoost ★ | 4.88 | 6.25 | 0.838 | Best |
| 24h | RandomForest | 5.56 | 7.17 | 0.787 | — |
| 24h | LinearRegression | 5.64 | 7.19 | 0.786 | — |
| 48h | XGBoost ★ | 5.49 | 7.17 | 0.787 | Best |
| 48h | LinearRegression | 5.71 | 7.35 | 0.776 | — |
| 48h | RandomForest | 6.39 | 8.23 | 0.719 | — |
| 72h | XGBoost ★ | 6.04 | 7.82 | 0.747 | Best |
| 72h | LinearRegression | 6.25 | 7.88 | 0.744 | — |
| 72h | RandomForest | 7.03 | 9.07 | 0.660 | — |

### 6.1 Why XGBoost Wins All Horizons After Full Dataset Extension

With only 6 months of data, LinearRegression won the 72h horizon because the signal was too weak for tree models to generalize. After extending to 16 months covering all four seasons, XGBoost gained sufficient training examples — particularly monsoon→winter transitions — to learn the non-linear seasonal effects that linear models cannot capture. This validated the data extension decision: more seasons improved tree model generalization specifically at the harder, longer horizons.

### 6.2 Most Important Features

SHAP (SHapley Additive exPlanations) was used to validate that models learned physically meaningful patterns:

**Day 1 (24h):** Short-lag features dominate — `aqi_lag_1h`, `aqi_lag_3h`, `aqi_rolling_mean_24h`. Current PM2.5 and current weather conditions appear in the top 10. The next 24h mean is primarily determined by where AQI is right now.

**Day 2 (48h):** Medium-lag features rise — `aqi_lag_24h`, `aqi_lag_36h` (the bridge anchor). Forecast features `temp_36h`, `wind_36h`, `pm2_5_36h` appear prominently, confirming that the t+36h bridge features fill the context gap exactly as designed.

**Day 3 (72h):** Long-lag and weather forecast features dominate — `aqi_lag_48h`, `aqi_lag_72h`, `temp_72h`, `wind_72h`, `pressure_72h`, `pm2_5_72h`. Short-lag features (correctly excluded via horizon-aware filtering) would have appeared near-zero if included, confirming the filtering made correct exclusions.

---

## 7. Challenges and Solutions

### 7.1 Infrastructure — Hopsworks → MongoDB Atlas Migration

#### Challenge

The original project plan called for using **Hopsworks** as both the feature store and model registry. Hopsworks provides a managed platform for storing feature groups, training datasets, and model artifacts — it is a clean, purpose-built solution for ML pipelines.

However, during development, Hopsworks was not working reliably. It intermittently returned `user not found` errors, and submitted jobs either took multiple days to complete or failed silently. This made Hopsworks inaccessible within the project timeline and required a pivot.

#### Solution

Migrated entirely to **MongoDB Atlas** (free tier, 512 MB) to handle both responsibilities:

- **Feature storage:** All ingested weather and air quality data is stored in MongoDB collections, with upsert logic on timestamps to avoid duplicates. This effectively acts as a feature store — the training pipeline reads directly from the defined collections.

- **Model registry:** Trained models are serialized with `pickle` and stored in MongoDB's **GridFS** (which handles files larger than the 16 MB document limit). Each model is stored with full metadata — scores, hyperparameters, training date, feature list, and a flag indicating whether it is the current best model.

This approach turned out to be simpler to maintain than Hopsworks would have been. MongoDB Atlas's free tier provides enough storage for this use case, and consolidating data + models into a single database reduces external dependencies from two services to one.

---

### 7.2 Target Variable — PM2.5 vs US AQI

#### Challenge

The initial approach was to predict **PM2.5 concentration (µg/m³)** as the target variable, then convert predictions to AQI using the standard EPA breakpoint formula. This seemed logical — PM2.5 is the raw measurement, and AQI is just a transformation of it.

The results were poor. Four compounding problems made this approach unworkable:

1. **Recursive lag error compounding.** PM2.5 autocorrelation is extremely strong. Once the model gets the first predicted hour slightly wrong, that error propagates into the lag features for the next prediction, compounding further with each step.

2. **Lag features drowning out everything else.** The model essentially learned *"PM2.5 tomorrow ≈ PM2.5 today"* — weather features (temperature, wind speed) and time features barely contributed. There was no meaningful multi-day forecasting happening.

3. **Error snowballing.** A small initial error of 5 µg/m³ could grow into a 30+ µg/m³ error by hour 72 through lag propagation alone — before the AQI conversion even applied.

4. **EPA breakpoint amplification.** Because of the breakpoint structure, a small PM2.5 error near a category boundary could cause the AQI to jump by 20–30 points, making an already-poor model look catastrophically wrong.

#### Solution

Changed the target variable from PM2.5 to **US AQI directly**. Instead of predicting the raw concentration and converting, the models predict the AQI value that Open-Meteo already computes using the official EPA formula:

```
AQI = ((I_high - I_low) / (C_high - C_low)) × (C - C_low) + I_low
```

This eliminated recursive error compounding entirely. AQI predictions are direct model outputs — no post-processing required — and models are evaluated on the same scale that the dashboard shows users.

---

### 7.3 Missing Seasons Caused Poor Multi-Day Performance

#### Challenge

The initial dataset covered **November 30, 2025 to May 18, 2026** — approximately 6 months. After training the first set of models, performance at the 48h and 72h horizons was noticeably poor. Investigation revealed the root cause: **seasonal coverage was incomplete**.

Hyderabad has four climatologically distinct seasons with dramatically different AQI profiles:

| Season | Months | Mean AQI | Atmospheric Driver |
|---|---|:---:|---|
| Winter | Dec, Jan, Feb | ~106 | Cold air inversions trap pollutants close to the ground |
| Pre-Monsoon | Mar, Apr, May | ~75 | Dry and windy — moderate dispersal |
| Monsoon | Jun, Jul, Aug, Sep | ~74 | Rainfall scavenges particulates from the atmosphere |
| Post-Monsoon | Oct, Nov | ~98 | Cooling air causes pollutant re-accumulation |

With only 6 months of data (Nov 2025 – May 2026), the models had **never seen the Monsoon season** (June–September), when AQI drops by 30+ points due to rainfall. This created a severe distribution gap — when predicting 48–72 hours ahead toward monsoon months, the model had no learned relationship between heavy rainfall patterns and AQI drop. Predictions were systematically and consistently too high.

#### Solution

Extended historical data back to **January 1, 2025** using Open-Meteo's historical archive API, giving the models a complete 16-month dataset covering all four seasons with adequate training examples of each.

| Attribute | Value |
|---|---|
| Date range | January 1, 2025 → May 30, 2026 |
| Total rows | 12,360 hourly records |
| Total days | ~515 days |
| Missing values | 0 |
| Duplicate records | 0 |
| Seasons covered | All 4 — Winter, Pre-Monsoon, Monsoon, Post-Monsoon |

This was the **single most impactful improvement** to multi-day forecast accuracy. The lesson: for seasonal forecasting problems, at least one complete annual cycle of training data is non-negotiable. Short-term data creates systematic bias in any direction where the model has no training examples.

---

## 8. Deployment

### 8.1 Infrastructure

| Component | Service | Cost |
|---|---|---|
| Dashboard | Streamlit Cloud | Free |
| Database | MongoDB Atlas (M0, 512 MB) | Free |
| CI/CD | GitHub Actions | Free (2,000 min/month) |
| Code | GitHub | Free |

**Total infrastructure cost: $0**

### 8.2 Dashboard Architecture

The dashboard reads exclusively from MongoDB — no model loading, no API calls, no feature computation at display time. All data is pre-computed by the live pipeline.

| Section | Source | Refresh |
|---|---|---|
| Current AQI + gauge | predictions.aqi_now | Hourly |
| Weather conditions | aqi_features | Hourly |
| Major pollutants | aqi_features | Hourly |
| 3-day forecast cards | predictions.aqi_24h/48h/72h | Hourly |
| 7-day trend chart | aqi_features (last 7 calendar days) | Hourly |
| Model performance | models collection | On retrain |
| Alerts (AQI > 150) | predictions | Hourly |

---

## 9. Limitations

- **Spatial resolution:** CAMS data represents an ~11 km grid cell, not a specific monitoring station. Actual AQI at any point in Hyderabad may differ from the grid average.
- **No local ground truth:** All training and evaluation is against CAMS-derived AQI values. There is no PM2.5 monitoring station in Hyderabad to independently validate against.
- **Occasional pipeline gaps:** GitHub Actions free tier sometimes delays or skips scheduled runs, creating 2–7 hour gaps in the feature store. Handled via manual gap fill but not fully preventable.
- **Open-Meteo forecast quality:** At t+72h, the Open-Meteo weather forecast itself has uncertainty. The model's accuracy is bounded by the quality of the forecast weather inputs it receives.
- **Single location:** System is configured for Hyderabad only. Expanding requires separate data pipelines and potentially different feature sets.

---

## 10. Future Improvements

- Integrate local ground monitoring station data when available in Hyderabad
- Add confidence intervals to predictions based on model uncertainty
- Add push notifications or webhook alerts when forecast AQI exceeds 150
- Expand to other cities in Pakistan — Karachi, Lahore, Islamabad
- Implement model drift detection to flag when new training produces significantly worse metrics
- Explore Temporal Fusion Transformer for improved long-range seasonal modeling

---

## 11. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10.11 |
| ML Models | XGBoost, RandomForest, LinearRegression (scikit-learn) |
| Data Source | Open-Meteo APIs |
| Database | MongoDB Atlas M0 free tier — features + model registry via GridFS |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions — Hourly & Daily Pipelines  |
| Hosting | Streamlit Cloud |
| Version Control | Git + GitHub |

---

## 12. Conclusion

This project demonstrates that a reliable, city-level multi-horizon AQI forecasting system can be built and deployed entirely on free-tier infrastructure using a direct multi-model architecture.

The key technical decisions that defined the project's success:

**Switching from PM2.5 to US AQI as the target variable** was the first critical decision. Predicting PM2.5 directly caused recursive error compounding through lag features, making forecasts unreliable beyond Day 1. Predicting AQI directly eliminated this problem entirely and produced clean, evaluable outputs.

**Migrating from Hopsworks to MongoDB Atlas** ensured the project could proceed when the originally planned infrastructure proved unreliable. Consolidating feature storage and model registry into a single database simplified the architecture and reduced external dependencies.

**Extending data from 6 to 16 months** was the single most impactful improvement to multi-day accuracy. Discovering that missing the Monsoon season caused systematic overestimation reinforced a fundamental principle: *seasonal forecasting requires at least one complete annual cycle of training data.*

**Separate model per horizon** avoided recursive error compounding that plagues single-model autoregressive approaches at 48h and 72h distances. Each model directly learns the relationship between current conditions and its specific future window.

**Horizon-aware feature filtering** prevented short-lag features from adding noise to longer-horizon models, improving generalization without any loss of 24h accuracy.

The system has been running autonomously since deployment — fetching live data every hour, making predictions for three days ahead, and serving results through a public dashboard — with zero manual intervention required.

---

- **Author:** Ahsan Ali
- **Data Source:** Open-Meteo
- **Organization:** 10Pearls — Internship Project
- **Repository:** https://github.com/Ahsanali18/10Pearls-Hyderabad-AQI-Predictor
- **Live Dashboard:** [View Live Application](https://10pearls-hyderabad-aqi-predictor-xgct4ppjp9s5qmyqucwxzv.streamlit.app/)