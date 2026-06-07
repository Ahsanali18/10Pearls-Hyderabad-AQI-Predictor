# 🌫️ Hyderabad AQI Prediction System

A production-grade machine learning system that forecasts the **US Air Quality Index (AQI) for Hyderabad, Sindh, Pakistan** across three daily horizons — Day 1 (next 24h), Day 2 (next 48h), and Day 3 (next 72h). The system runs fully autonomously on free-tier infrastructure, updating predictions every hour without any manual intervention.

🔗 **[View Live Application](https://10pearls-hyderabad-aqi-predictor-xgct4ppjp9s5qmyqucwxzv.streamlit.app/)**
---
<img width="1654" height="756" alt="image" src="https://github.com/user-attachments/assets/bf202677-a02e-4fdd-ae48-a91394f6b137" />


---

## Why This Exists

Hyderabad regularly experiences moderate to unhealthy air quality driven by vehicular traffic, industrial emissions, dust storms, and seasonal temperature inversions. No localized multi-day AQI forecast existed for the city — existing services either focus on Karachi or only show current readings. This system fills that gap with a city-specific, three-day ahead forecast served through a public dashboard.

---

## What It Does

- Fetches hourly weather and air quality data from **Open-Meteo APIs** (archive, forecast, and current)
- Engineers **60 features** across 11 categories — lag features, rolling statistics, cyclical time encodings, future weather anchors, and physics-derived interaction terms
- Trains **3 separate XGBoost models** (one per horizon) using horizon-aware feature filtering
- Stores all features and model artifacts in **MongoDB Atlas**
- Serves a live **Streamlit dashboard** showing current AQI, pollutant breakdown, and 3-day forecast cards
- Runs on a **GitHub Actions schedule** — ingestion every hour, retraining daily — zero human intervention required

---

## Model Performance

Three models are trained daily (XGBoost, RandomForest, LinearRegression) and evaluated per horizon. The best model is selected dynamically based on the lowest RMSE. 

| Horizon | Model | MAE | RMSE | R² | Status |
|---|---|---|---|---|---|
| 24h | XGBoost ★ | 4.88 | 6.25 | 0.838 | Best |
| 24h | RandomForest | 5.56 | 7.17 | 0.787 | — |
| 24h | LinearRegression | 5.64 | 7.19 | 0.786 | — |
| 48h | XGBoost ★ | 5.49 | 7.17 | 0.787 | Best |
| 48h | LinearRegression | 5.71 | 7.35 | 0.776 | — |
| 48h | RandomForest | 6.39 | 8.23 | 0.719 | — |
| 72h | XGBoost ★ | 6.04 | 7.82 | 0.746 | Best |
| 72h | LinearRegression | 5.95 | 7.58 | 0.762 | — |
| 72h | RandomForest | 6.96 | 9.03 | 0.662 | — |
 

> These scores reflect the latest production training run. Since training happens daily with fresh data, exact values may shift slightly over time. Each horizon has its own dedicated model — no autoregressive error compounding across days.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                           │
│   ┌──────────────────────┐    ┌──────────────────────────────┐  │
│   │  Hourly Ingestion    │    │     Daily Retraining         │  │
│   │  backfill_pipeline   │    │     training_pipeline        │  │
│   └──────────┬───────────┘    └──────────────┬───────────────┘  │
└──────────────┼───────────────────────────────┼──────────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐     ┌─────────────────────────────┐
│     Open-Meteo APIs      │     │       MongoDB Atlas          │
│  - Archive API           │────▶│  - aqi_features collection  │
│  - Forecast API          │     │  - weather_forecast         │
│  - Air Quality API       │     │  - predictions              │
└──────────────────────────┘     │  - models (GridFS)          │
                                 └──────────────┬──────────────┘
                                                │
               ┌────────────────────────────────┤
               │                                │
               ▼                                ▼
┌──────────────────────────┐     ┌─────────────────────────────┐
│    Live Pipeline         │     │     Streamlit Dashboard      │
│  - Feature engineering   │     │  - Current AQI gauge        │
│  - Horizon-aware filter  │     │  - 3-day forecast cards     │
│  - XGBoost inference     │     │  - Pollutant breakdown      │
│  - Write predictions     │     │  - 7-day trend chart        │
└──────────────────────────┘     └─────────────────────────────┘
```

**Data flow at inference:**
1. GitHub Actions triggers `live_pipeline.py` every hour
2. Live pipeline fetches current conditions + Open-Meteo forecast
3. Features are computed from live data + last 72 rows stored in MongoDB
4. All Three  models produce Day 1, Day 2, Day 3 predictions
5. Predictions are written to MongoDB
6. Dashboard reads from MongoDB and renders to display live results.

---

## Project Structure

```
HYDERABAD_AQI_PREDICTION/
│
├── src/
│   ├── data_fetching/
│   │   └── fetch_data.py              # Open-Meteo archive, forecast & air quality API calls
│   │
│   ├── database/
│   │   ├── database_connection.py     # MongoDB Atlas connection
│   │   └── model_registry.py         # GridFS model storage + metadata (save/load best models)
│   │
│   ├── features/
│   │   └── feature_engineering.py    # 60-feature pipeline — lags, rolling stats,
│   │                                 # cyclical encodings, future weather anchors,
│   │                                 # horizon-aware filtering, live feature computation
│   │
│   └── pipelines/
│       ├── backfill_pipeline.py      # One-time historical data backfill (Jan 2025 → present)
│       ├── live_pipeline.py          # Hourly: fetch → engineer features → predict → store
│       └── training_pipeline.py      # Daily: load features → train 9 models → save best
│
├── config/
│   └── settings.py                   # Coordinates, API endpoints, DB collection names,
│                                     # horizon config, feature group definitions
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard
│                                     
│
├── notebooks/
│   └── EDA.ipynb                     # Exploratory data analysis — AQI distribution,
│                                     # autocorrelation, weather correlations, seasonal patterns
│
├── .github/
│   └── workflows/
│       ├── hourly_ingestion.yml      # Triggers live_pipeline.py every hour
│       └── daily_training.yml        # Triggers training_pipeline.py every day
│
├── .streamlit/
│   └── config.toml                   # Dashboard theme configuration
│
├── .env                              # MongoDB URI, API keys (not committed)
├── .gitignore
├── requirements.txt
├── Report.md                         # Full technical project report
└── README.md
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Ahsanali18/10Pearls-Hyderabad-AQI-Predictor.git
cd 10Pearls-Hyderabad-AQI-Predictor

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGODB_URI="mongodb+srv://..."

# Backfill historical data (run once)
python -m src.pipelines.backfill_pipeline

# Train models
python -m src.pipelines.training_pipeline

# Run live pipeline (single pass)
python -m src.pipelines.live_pipeline

# Launch dashboard
streamlit run dashboard/app.py
```

---

## CI/CD — GitHub Actions

Both workflows use `MONGODB_URI` from GitHub Secrets.

| Workflow | Schedule | What It Does |
|---|---|---|
| Hourly Ingestion | Every hour at :05 | Fetches latest weather + AQI data from Open-Meteo, engineers features, runs inference, upserts predictions to MongoDB |
| Daily Training | 2:00 AM UTC daily | Loads full feature history, trains 9 models (3 algorithms × 3 horizons), saves best per horizon to GridFS |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10.11 |
| ML Models | XGBoost, RandomForest, LinearRegression (scikit-learn) |
| Data Source | Open-Meteo APIs (Archive, Forecast, Air Quality) |
| Database | MongoDB Atlas M0 free tier — features + model registry via GridFS |
| Dashboard | Streamlit + Plotly |
| CI/CD | GitHub Actions |
| Hosting | Streamlit Cloud |
| Version Control | Git + GitHub |

**Total infrastructure cost: $0**

---

---

## Author: **Ahsan Ali**
## Organization: 10Pearls Internship Project
· 📄 [Full Project Report](./Report.md)

*Data source: Open-Meteo · Location: 25.396°N, 68.358°E — Hyderabad, Sindh, Pakistan*
