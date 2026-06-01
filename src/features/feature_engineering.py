"""
src/features/feature_engineering.py
=====================================
Hyderabad AQI — Feature Engineering

TARGET DEFINITION — daily mean AQI:
    aqi_24h = mean AQI over t+1  → t+24   (Day 1)
    aqi_48h = mean AQI over t+25 → t+48   (Day 2)
    aqi_72h = mean AQI over t+49 → t+72   (Day 3)

WHY DAY 2 WAS STILL WEAK (R²=0.449):
─────────────────────────────────────────────────────
    Day 2 sits in a middle-horizon gap:
    - Lag anchors (lag_48h r=0.570) are weaker than Day 1's lag_24h (r=0.743)
    - Future weather at t+48 alone is not enough context
    - The model had no direct signal about what AQI will be at t+24
      (the bridge between Day 1 and Day 2)

THE FIX — bridge features for Day 2:
─────────────────────────────────────────────────────
    1. aqi_36h_lag   — AQI at t-36h  (midpoint anchor between 24h and 48h)
    2. future weather at t+36h  — midpoint between Day1 and Day2 horizon
    3. aqi_day1_pred_proxy — rolling mean of t+1..t+24 window using
       available past data (gives the model an estimate of Day1 AQI
       which is the strongest predictor of Day2 AQI)

FEATURE GROUPS (55 total):
─────────────────────────────────────────────────────────────────
Group                        Count
─────────────────────────────────────────────────────────────────
Weather current               (8)
Pollutants current            (6)
Future weather t+24           (5)   temp/wind/pressure/humidity/pm2_5
Future weather t+36           (5)   bridge features for Day 2
Future weather t+48           (5)
Future weather t+72           (5)
Cyclical Time                 (6)
Time Flags                    (5)
Lag Features                  (8)   added lag_36h as Day2 bridge anchor
Rolling Statistics            (3)
Derived                       (4)
─────────────────────────────────────────────────────────────────
TOTAL                        55
"""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [

    # Current weather (8)
    "temperature",
    "humidity",
    "rain",
    "wind_speed",
    "pressure",
    "weather_code",
    "wind_dir_sin",
    "wind_dir_cos",

    # Current pollutants (6)
    "pm2_5",
    "so2",
    "co",
    "no2",
    "pm10",
    "o3",

    # Future weather at t+24 (5)
    # Gives Day 1 horizon context, also helps Day 2 as a near anchor
    "temp_24h",
    "wind_24h",
    "pressure_24h",
    "humidity_24h",
    "pm2_5_24h",

    # Future weather at t+36 (5)
    # Bridge features, midpoint between Day 1 and Day 2 horizons
    # Directly fills the context gap that caused Day 2 weakness
    "temp_36h",
    "wind_36h",
    "pressure_36h",
    "humidity_36h",
    "pm2_5_36h",

    # Future weather at t+48 (5)
    "temp_48h",
    "wind_48h",
    "pressure_48h",
    "humidity_48h",
    "pm2_5_48h",

    # Future weather at t+72 (5)
    "temp_72h",
    "wind_72h",
    "pressure_72h",
    "humidity_72h",
    "pm2_5_72h",

    # Cyclical Time (6)
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",

    # Time Flags (5)
    "is_night",
    "is_morning_rush",
    "is_afternoon",
    "is_evening_rush",
    "is_weekend",

    # Lag Features (8)
    # lag_36h added as midpoint anchor specifically for Day 2
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_lag_36h",      
    "aqi_lag_48h",
    "aqi_lag_72h",

    # Rolling Statistics (3)
    "aqi_rolling_mean_24h",
    "aqi_rolling_std_24h",
    "aqi_rolling_mean_6h",

    # Derived / Interaction (4)
    "aqi_change_rate",
    "dispersion_index",
    "heat_humidity",
    "pressure_change",
]

TARGET_COLUMNS = [
    "aqi_24h",
    "aqi_48h",
    "aqi_72h",
]


def _add_targets(df):
    """Daily mean targets — average AQI over each 24h window."""
    df  = df.copy()
    aqi = df["aqi"].values
    n   = len(aqi)

    day1 = np.full(n, np.nan)
    day2 = np.full(n, np.nan)
    day3 = np.full(n, np.nan)

    for i in range(n):
        if i + 24 < n:
            day1[i] = np.mean(aqi[i + 1  : i + 25])
        if i + 48 < n:
            day2[i] = np.mean(aqi[i + 25 : i + 49])
        if i + 72 < n:
            day3[i] = np.mean(aqi[i + 49 : i + 73])

    df["aqi_24h"] = day1
    df["aqi_48h"] = day2
    df["aqi_72h"] = day3
    return df


def _add_future_weather_features(df):
    """
    Future weather features at t+24, t+36, t+48, t+72.

    Training  : historical values shifted backward using shift(-N)
    Inference : fetched from Open-Meteo forecast API for each horizon
    """

    df = df.copy()

    for h in [24, 36, 48, 72]:
        prefix = f"_{h}h"
        df[f"temp{prefix}"]     = df["temperature"].shift(-h)
        df[f"wind{prefix}"]     = df["wind_speed"].shift(-h)
        df[f"pressure{prefix}"] = df["pressure"].shift(-h)
        df[f"humidity{prefix}"] = df["humidity"].shift(-h)
        df[f"pm2_5{prefix}"]    = df["pm2_5"].shift(-h)

    return df


def _add_time_features(df):
    df = df.copy()
    hour  = df.index.hour
    month = df.index.month
    dow   = df.index.dayofweek

    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)

    df["is_night"]        = ((hour >= 22) | (hour < 6)).astype(int)
    df["is_morning_rush"] = ((hour >= 6)  & (hour < 10)).astype(int)
    df["is_afternoon"]    = ((hour >= 11) & (hour < 17)).astype(int)
    df["is_evening_rush"] = ((hour >= 17) & (hour < 21)).astype(int)
    df["is_weekend"]      = (dow >= 5).astype(int)
    return df


def _add_lag_features(df):
    df = df.copy()
    for h in [1, 3, 6, 12, 24, 36, 48, 72]:
        df[f"aqi_lag_{h}h"] = df["aqi"].shift(h)
    return df


def _add_rolling_features(df):
    df = df.copy()
    past_aqi = df["aqi"].shift(1)
    df["aqi_rolling_mean_24h"] = past_aqi.rolling(window=24, min_periods=6).mean()
    df["aqi_rolling_std_24h"]  = past_aqi.rolling(window=24, min_periods=6).std()
    df["aqi_rolling_mean_6h"]  = past_aqi.rolling(window=6,  min_periods=2).mean()
    return df


def _add_derived_features(df):
    df = df.copy()
    wind_rad           = np.deg2rad(df["wind_direction"])
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)
    df["aqi_change_rate"]  = df["aqi"].diff(1)
    df["dispersion_index"] = df["wind_speed"] * df["temperature"]
    df["heat_humidity"]    = df["temperature"] * df["humidity"]
    df["pressure_change"]  = df["pressure"].diff(1)
    return df


def engineer_features(df: pd.DataFrame):
    """
    Full feature engineering for historical/backfill data.
    Input:  raw DataFrame with DatetimeIndex.
    Output: 56 FEATURE_COLUMNS + 3 TARGET_COLUMNS + aqi.
    """
    df = df.copy()
    df.sort_index(inplace=True)

    df = _add_targets(df)
    df = _add_future_weather_features(df)
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_derived_features(df)

    keep   = FEATURE_COLUMNS + TARGET_COLUMNS + ["aqi"]
    df     = df[keep]
    before = len(df)
    df     = df.dropna()

    print(f"[FE] Rows    : {before:,} → {len(df):,}  (dropped {before - len(df):,})")
    print(f"[FE] Shape   : {df.shape}")
    print(f"[FE] Features: {len(FEATURE_COLUMNS)}  |  Targets: {len(TARGET_COLUMNS)}")
    for t in TARGET_COLUMNS:
        print(f"[FE] {t}: mean={df[t].mean():.1f}  std={df[t].std():.1f}")

    return df


def compute_live_features(current_row: dict, history_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """
    Compute all 55 features for one live row at inference time.

    Args:
        current_row  : dict — current hour's sensor values
        history_df   : last 72 rows from MongoDB (oldest → newest)
                       Required columns: aqi, pressure
        forecast_df  : DataFrame from Open-Meteo forecast API (next 5 days)
                       Index: DatetimeIndex (hourly)
                       Required columns: temperature, wind_speed, pressure,
                                         humidity, pm2_5
    """
    ts    = pd.Timestamp(current_row["time"])
    aqi   = float(current_row["aqi"])
    hour  = ts.hour
    month = ts.month
    dow   = ts.dayofweek

    # Time
    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin   = np.sin(2 * np.pi * dow   / 7)
    dow_cos   = np.cos(2 * np.pi * dow   / 7)

    is_night        = int(hour >= 22 or hour < 6)
    is_morning_rush = int(6  <= hour < 10)
    is_afternoon    = int(11 <= hour < 17)
    is_evening_rush = int(17 <= hour < 21)
    is_weekend      = int(dow >= 5)

    # Lags
    past_aqi = history_df["aqi"].values
    def get_lag(h):
        idx = len(past_aqi) - h
        return float(past_aqi[idx]) if idx >= 0 else float(np.nan)

    # Rolling
    aqi_rolling_mean_24h = float(np.mean(past_aqi[-24:])) if len(past_aqi) >= 6  else float(np.nan)
    aqi_rolling_std_24h  = float(np.std(past_aqi[-24:], ddof=1)) if len(past_aqi) >= 6 else float(np.nan)
    aqi_rolling_mean_6h  = float(np.mean(past_aqi[-6:]))  if len(past_aqi) >= 2 else float(np.nan)

    # Derived
    prev_aqi      = float(past_aqi[-1]) if len(past_aqi) >= 1 else aqi
    wind_rad      = np.deg2rad(float(current_row["wind_direction"]))
    prev_pressure = float(history_df["pressure"].iloc[-1]) if len(history_df) >= 1 else float(current_row["pressure"])

    # Future weather from forecast API
    def get_forecast(col, hours_ahead):
        target_ts = ts + pd.Timedelta(hours=hours_ahead)
        if forecast_df is not None and col in forecast_df.columns:
            diff = np.abs((forecast_df.index - target_ts).total_seconds().to_numpy())
            idx  = int(np.argmin(diff))
            if diff[idx] <= 3600:
                return float(forecast_df[col].iloc[idx])
        return float(np.nan)


    return {
        "temperature":          float(current_row["temperature"]),
        "humidity":             float(current_row["humidity"]),
        "rain":                 float(current_row["rain"]),
        "wind_speed":           float(current_row["wind_speed"]),
        "pressure":             float(current_row["pressure"]),
        "weather_code":         int(current_row["weather_code"]),
        "wind_dir_sin":         float(np.sin(wind_rad)),
        "wind_dir_cos":         float(np.cos(wind_rad)),
        "pm2_5":                float(current_row["pm2_5"]),
        "so2":                  float(current_row["so2"]),
        "co":                   float(current_row["co"]),
        "no2":                  float(current_row["no2"]),
        "pm10":                 float(current_row["pm10"]),
        "o3":                   float(current_row["o3"]),
        "temp_24h":             get_forecast("temperature", 24),
        "wind_24h":             get_forecast("wind_speed",  24),
        "pressure_24h":         get_forecast("pressure",    24),
        "humidity_24h":         get_forecast("humidity",    24),
        "pm2_5_24h":            get_forecast("pm2_5",       24),
        "temp_36h":             get_forecast("temperature", 36),
        "wind_36h":             get_forecast("wind_speed",  36),
        "pressure_36h":         get_forecast("pressure",    36),
        "humidity_36h":         get_forecast("humidity",    36),
        "pm2_5_36h":            get_forecast("pm2_5",       36),
        "temp_48h":             get_forecast("temperature", 48),
        "wind_48h":             get_forecast("wind_speed",  48),
        "pressure_48h":         get_forecast("pressure",    48),
        "humidity_48h":         get_forecast("humidity",    48),
        "pm2_5_48h":            get_forecast("pm2_5",       48),
        "temp_72h":             get_forecast("temperature", 72),
        "wind_72h":             get_forecast("wind_speed",  72),
        "pressure_72h":         get_forecast("pressure",    72),
        "humidity_72h":         get_forecast("humidity",    72),
        "pm2_5_72h":            get_forecast("pm2_5",       72),
        "hour_sin":             hour_sin,
        "hour_cos":             hour_cos,
        "month_sin":            month_sin,
        "month_cos":            month_cos,
        "dow_sin":              dow_sin,
        "dow_cos":              dow_cos,
        "is_night":             is_night,
        "is_morning_rush":      is_morning_rush,
        "is_afternoon":         is_afternoon,
        "is_evening_rush":      is_evening_rush,
        "is_weekend":           is_weekend,
        "aqi_lag_1h":           get_lag(1),
        "aqi_lag_3h":           get_lag(3),
        "aqi_lag_6h":           get_lag(6),
        "aqi_lag_12h":          get_lag(12),
        "aqi_lag_24h":          get_lag(24),
        "aqi_lag_36h":          get_lag(36),
        "aqi_lag_48h":          get_lag(48),
        "aqi_lag_72h":          get_lag(72),
        "aqi_rolling_mean_24h": aqi_rolling_mean_24h,
        "aqi_rolling_std_24h":  aqi_rolling_std_24h,
        "aqi_rolling_mean_6h":  aqi_rolling_mean_6h,
        "aqi_change_rate":      aqi - prev_aqi,
        "dispersion_index":     float(current_row["wind_speed"]) * float(current_row["temperature"]),
        "heat_humidity":        float(current_row["temperature"]) * float(current_row["humidity"]),
        "pressure_change":      float(current_row["pressure"]) - prev_pressure,
        "aqi_24h":              float("nan"),
        "aqi_48h":              float("nan"),
        "aqi_72h":              float("nan"),
        "aqi":                  aqi,
    }