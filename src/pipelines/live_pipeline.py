"""
Hyderabad AQI — Live Inference Pipeline Module: Runs every hour via GitHub Actions.
"""

import sys
from datetime import datetime, timezone

import pytz
import requests
import pandas as pd
import numpy as np

from config.settings import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    MONGO_COLLECTION,
    MONGO_MODELS_COLLECTION
)
from src.data_fetching.fetch_data import COLUMN_MAPPING
from src.features.feature_engineering import compute_live_features, TARGET_COLUMNS, get_features
from src.database.database_connection import get_db_client, nan_to_none
from src.database.model_registry import load_model
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30), reraise=True)
def _safe_get(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

WEATHER_FORECAST_URL     = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_FORECAST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
PREDICTIONS_COLLECTION   = "predictions"

REQUIRED_LAG_HOURS = [1, 3, 6, 12, 24, 36, 48, 72]
MIN_HISTORY_ROWS   = 72    # below this, Day 2/3 predictions degrade


# AQI Category Helper
AQI_CATEGORIES = [
    (50,  "Good",                           "🟢"),
    (100, "Moderate",                        "🟡"),
    (150, "Unhealthy for Sensitive Groups",  "🟠"),
    (200, "Unhealthy",                       "🔴"),
    (300, "Very Unhealthy",                  "🟣"),
    (999, "Hazardous",                       "🟤"),
]

def aqi_label(value):
    """Return (category_string, emoji) for a given AQI value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown", "⚪"
    for threshold, label, emoji in AQI_CATEGORIES:
        if value <= threshold:
            return label, emoji
    return "Hazardous", "🟤"


# STEP 1 — Fetch Current Hour
def fetch_current_hour():
    """
    Fetches current hour weather + air quality from OpenMeteo forecast API.

    Uses past_days=1 + forecast_days=1 so the current hour is always
    included regardless of what time the pipeline runs.

    Falls back to last available hour if the current hour has no AQI yet
    (OpenMeteo air quality sometimes lags ~1h behind real time).

    Returns:
        dict with renamed columns matching feature_engineering.py expectations:
        time, temperature, humidity, rain, wind_speed, wind_direction,
        pressure, weather_code, pm2_5, pm10, no2, so2, o3, co, aqi
    """
    print("\n[1/6] Fetching current hour from OpenMeteo ...")

    karachi_tz   = pytz.timezone(TIMEZONE)
    current_time = datetime.now(karachi_tz)
    current_hour = current_time.strftime("%Y-%m-%dT%H:00")
    print(f"        Current local hour : {current_hour}")

    # Weather 
    weather_params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "weather_code",
        ],
        "timezone":      TIMEZONE,
        "past_days":     1,      # ensures current hour is always present
        "forecast_days": 1,
    }

    try:
        weather_data = _safe_get(WEATHER_FORECAST_URL, weather_params)
    except Exception as e:
        print(f"[ERROR] Weather fetch failed: {e}")
        sys.exit(1)

    # Air Quality
    aq_params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly": [
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "carbon_monoxide",
            "us_aqi",
        ],
        "timezone":      TIMEZONE,
        "past_days":     1,    
        "forecast_days": 1,
    }

    try:
        aq_data = _safe_get(AIR_QUALITY_FORECAST_URL, aq_params)
    except Exception as e:
        print(f"[ERROR] Air quality fetch failed: {e}")
        sys.exit(1)

    # Merge and rename
    weather_df = pd.DataFrame(weather_data["hourly"])
    aq_df      = pd.DataFrame(aq_data["hourly"])
    merged_df  = pd.merge(weather_df, aq_df, on="time")
    merged_df.rename(columns=COLUMN_MAPPING, inplace=True)

    # Pick current hour, fall back to last available 
    current_df = merged_df[merged_df["time"] == current_hour]

    if current_df.empty or current_df["aqi"].isna().all():
        available_df = merged_df.dropna(subset=["aqi"])
        if available_df.empty:
            print("[ERROR] No AQI data available from OpenMeteo. Aborting.")
            sys.exit(1)
        current_row = available_df.iloc[-1].to_dict()
        print(f"        [WARN] Current hour not available — using last available hour")
    else:
        current_row = current_df.iloc[-1].to_dict()

    label, emoji = aqi_label(float(current_row["aqi"]))
    print(f"        Fetched hour  : {current_row['time']}")
    print(f"        AQI           : {current_row['aqi']:.0f}  {emoji} {label}")
    print(f"        PM2.5         : {current_row.get('pm2_5', 'N/A')} µg/m³")
    print(f"        Temperature   : {current_row.get('temperature', 'N/A')} °C")
    print(f"        Wind Speed    : {current_row.get('wind_speed', 'N/A')} km/h")

    return current_row


# STEP 2 — Fetch 4-Day Forecast 
def fetch_forecast_df():
    """
    Fetches 4-day hourly weather + PM2.5 forecast from OpenMeteo.
    4 days covers t+72 (Day 3) and t+36 bridge features comfortably.
    Returns:
        DataFrame with DatetimeIndex and columns:
        temperature, wind_speed, pressure, humidity, pm2_5
    """
    print("\n[2/6] Fetching 4-day forecast for future weather features ...")

    # Weather forecast 
    weather_params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure",
        ],
        "timezone":      TIMEZONE,
        "forecast_days": 4,
    }

    try:
        weather_data = _safe_get(WEATHER_FORECAST_URL, weather_params)
    except Exception as e:
        print(f"        [WARN] Forecast weather fetch failed: {e}")
        print(f"        Future weather features will be NaN — Day 2/3 predictions may degrade.")
        return None

    # Air quality forecast
    aq_params = {
        "latitude":      LATITUDE,
        "longitude":     LONGITUDE,
        "hourly":        ["pm2_5"],
        "timezone":      TIMEZONE,
        "forecast_days": 4,
    }

    try:
        aq_data = _safe_get(AIR_QUALITY_FORECAST_URL, aq_params)
    except Exception as e:
        print(f"        [WARN] Forecast Air Quality fetch failed: {e}")
        return None

    # Build forecast DataFrame 
    weather_df  = pd.DataFrame(weather_data["hourly"])
    aq_df       = pd.DataFrame(aq_data["hourly"])
    forecast_df = pd.merge(weather_df, aq_df, on="time")

    forecast_df.rename(columns={
        "temperature_2m":       "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m":       "wind_speed",
        "surface_pressure":     "pressure",
    }, inplace=True)

    forecast_df["time"] = pd.to_datetime(forecast_df["time"])
    forecast_df.set_index("time", inplace=True)

    print(f"        Forecast rows  : {len(forecast_df)}")
    print(f"        Range          : {forecast_df.index[0]}  →  {forecast_df.index[-1]}")
    print(f"        Columns        : {list(forecast_df.columns)}")

    return forecast_df


# STEP 3 — Fetch History from MongoDB
def fetch_history(col):
    """
    Reads last 72 rows from MongoDB features collection (oldest → newest).
    Returns:
        DataFrame sorted oldest → newest with columns: aqi, pressure, timestamp
    """
    print("\n[3/6] Reading last 72 rows from MongoDB ...")

    docs = list(
        col.find(
            {},
            {"aqi": 1, "pressure": 1, "timestamp": 1, "_id": 0}
        )
        .sort("timestamp", -1)
        .limit(72)
    )

    history_df = pd.DataFrame(docs).sort_values("timestamp").reset_index(drop=True)
    n_rows     = len(history_df)

    print(f"        History rows : {n_rows}")

    # Warn specifically about which lag features will be affected
    if n_rows < MIN_HISTORY_ROWS:
        print(f"        [WARN] Only {n_rows}/72 rows available.")
        for lag_h in REQUIRED_LAG_HOURS:
            status = "✓" if n_rows >= lag_h else f"✗ MISSING — will be 0 (hurts Day {'2/3' if lag_h >= 36 else '1'})"
            print(f"               aqi_lag_{lag_h}h : {status}")
    else:
        print(f"        [OK] Full 72-row history — all lag features populated")
        print(f"        Range : {history_df['timestamp'].iloc[0]}  →  {history_df['timestamp'].iloc[-1]}")

    return history_df


# STEP 4 — Compute Features
def build_feature_row(current_row, history_df, forecast_df):

    print("\n[4/6] Computing features ...")

    feature_dict = compute_live_features(current_row, history_df, forecast_df)

    feature_dict["timestamp"] = pd.Timestamp(current_row["time"])

    # Feature quality report 
    print(f"        Timestamp        : {feature_dict['timestamp']}")
    print(f"        Current AQI      : {feature_dict.get('aqi', 'N/A'):.1f}")
    print()

    # Lag features — critical for all 3 days
    print(f"        ── Lag features (critical for prediction quality) ──")
    for h in REQUIRED_LAG_HOURS:
        key = f"aqi_lag_{h}h"
        val = feature_dict.get(key, np.nan)
        day_note = "(Day2 bridge)" if h == 36 else "(Day3 bridge)" if h in [48, 72] else ""
        status   = f"{val:.1f}" if not np.isnan(val) else "NaN ← will be 0-filled, degrades predictions"
        print(f"        {key:<18} : {status}  {day_note}")

    # Future weather — needed for temp_24h, wind_24h, pm2_5_24h etc.
    print(f"\n        ── Future weather features ──")
    for h in [24, 36, 48, 72]:
        temp_key = f"temp_{h}h"
        pm_key   = f"pm2_5_{h}h"
        temp_val = feature_dict.get(temp_key, np.nan)
        pm_val   = feature_dict.get(pm_key,   np.nan)
        temp_str = f"{temp_val:.1f}" if not np.isnan(temp_val) else "NaN"
        pm_str   = f"{pm_val:.1f}"   if not np.isnan(pm_val)   else "NaN"
        print(f"        t+{h}h  temp={temp_str:>6}°C   pm2_5={pm_str:>6} µg/m³")

    return feature_dict


# STEP 5 — Upsert Feature Row to MongoDB 
def upsert_feature_row(col, feature_dict):
    """
    Upserts the current hour's complete feature row to MongoDB.
    Uses timestamp as unique key.
    """
    print("\n[5/6] Upserting feature row to MongoDB ...")

    ts  = feature_dict["timestamp"]
    doc = {"timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc)}

    skip_keys = {"timestamp", "aqi_24h", "aqi_48h", "aqi_72h"}   # targets not stored here

    for key, val in feature_dict.items():
        if key in skip_keys:
            continue
        if isinstance(val, np.integer):
            doc[key] = int(val)
        elif isinstance(val, (np.floating, float)):
            doc[key] = nan_to_none(float(val))
        else:
            doc[key] = val

    col.update_one(
        {"timestamp": doc["timestamp"]},
        {"$set": doc},
        upsert=True,
    )

    print(f"        Upserted row for : {doc['timestamp']}")


# STEP 6 — Predict and Store 
def _model_exists(db, target):
    """
    Check if a trained model exists in MongoDB for the given target.
    """
    models_col = db[MONGO_MODELS_COLLECTION]
    return models_col.find_one({"target":  target, "is_best": True}) is not None


def predict_and_store(feature_dict, db):
    """
    Loads best model per target from MongoDB GridFS and predicts.
    Stores one prediction document per timestamp in predictions collection:
        timestamp, made_at, aqi_now, aqi_24h, aqi_48h, aqi_72h
    """
    HORIZON_MAP = {
        "aqi_24h": "24h",
        "aqi_48h": "48h",
        "aqi_72h": "72h",
        }



    print("\n[6/6] Loading models and making predictions ...")

    if not _model_exists(db, "aqi_24h"):
        print("        [SKIP] No trained models in GridFS.")
        print("               Run training_pipeline.py first, then re-run live pipeline.")
        return {}

    predictions = {}
    pred_col    = db[PREDICTIONS_COLLECTION]
    made_at     = datetime.now(timezone.utc)

    for target in TARGET_COLUMNS:
        bundle     = load_model(db, target)
        model      = bundle["model"]
        scaler     = bundle["scaler"]
        model_name = bundle["model_name"]
        horizon   = HORIZON_MAP[target]  #different features for each horizon
        feat_cols = get_features(horizon) 

        # Build feature vector using exact columns this model was trained on
        # Use .get() with np.nan fallback — safer than direct dict access
        raw_values = np.array(
            [feature_dict.get(col, np.nan) for col in feat_cols],
            dtype=float
        ).reshape(1, -1)

        # NaN fill with 0 as last resort — log how many were affected
        nan_count = np.isnan(raw_values).sum()
        if nan_count > 0:
            nan_cols = [feat_cols[i] for i in range(len(feat_cols)) if np.isnan(raw_values[0, i])]
            print(f"        [WARN] {target} — {nan_count} NaN features zero-filled:")
            for c in nan_cols:
                print(f"               • {c}")
            raw_values = np.nan_to_num(raw_values, nan=0.0)

        # Apply scaler ONLY for LinearRegression — tree models are scale-invariant
        if model_name == "LinearRegression":
            X = scaler.transform(raw_values)
        else:
            X = raw_values   # XGBoost / RandomForest

        pred = float(model.predict(X)[0])
        pred = round(max(0.0, pred), 2)   # AQI cannot be negative

        predictions[target] = pred
        label, emoji = aqi_label(pred)
        print(f"        {target:<12} → {pred:>6.1f}  {emoji} {label}  (model: {model_name})")

    # Store prediction document 
    ts  = feature_dict["timestamp"]
    doc = {
        "timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc),
        "made_at":   made_at,
        "aqi_now":   float(feature_dict["aqi"]),
        "aqi_24h":   predictions.get("aqi_24h"),
        "aqi_48h":   predictions.get("aqi_48h"),
        "aqi_72h":   predictions.get("aqi_72h"),
    }

    pred_col.update_one(
        {"timestamp": doc["timestamp"]},
        {"$set": doc},
        upsert=True,
    )

    print(f"\n        Stored prediction document for {doc['timestamp']}")
    return predictions


# Main 
def run_live_pipeline():
    print("\n" + "=" * 60)
    print("LIVE PIPELINE — Current Hour → Features → Predict → Store")
    print("=" * 60)

    client, db = get_db_client()
    col        = db[MONGO_COLLECTION]

    try:
        current_row  = fetch_current_hour()
        forecast_df  = fetch_forecast_df()
        history_df   = fetch_history(col)
        feature_dict = build_feature_row(current_row, history_df, forecast_df)
        upsert_feature_row(col, feature_dict)
        predictions  = predict_and_store(feature_dict, db)

        # Final summary
        print("\n" + "=" * 60)
        print("LIVE PIPELINE COMPLETE")
        print("=" * 60)

        cur_label, cur_emoji = aqi_label(float(feature_dict["aqi"]))
        print(f"  Current AQI  : {feature_dict['aqi']:.0f}  {cur_emoji} {cur_label}")

        if predictions:
            horizons = {
                "aqi_24h": "Day 1 (~24h)",
                "aqi_48h": "Day 2 (~48h)",
                "aqi_72h": "Day 3 (~72h)",
            }
            for key, label in horizons.items():
                val = predictions.get(key)
                if val is not None:
                    cat, emoji = aqi_label(val)
                    print(f"  {label:<14} : {val:>6.1f}  {emoji} {cat}")
        else:
            print("  Predictions  : skipped — no trained models found")

        print("=" * 60)

    except Exception as e:
        print(f"\n[FATAL] Live pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    run_live_pipeline()