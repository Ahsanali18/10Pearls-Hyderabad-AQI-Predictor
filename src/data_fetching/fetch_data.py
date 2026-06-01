import os
import json
import requests
import pandas as pd

from datetime import datetime, timedelta

from config.settings import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    RAW_DATA_DIR,
    MERGED_JSON_PATH,
)


# API ENDPOINTS
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# VARIABLES TO FETCH
WEATHER_VARIABLES = [
    "temperature_2m",            # Cold temps trap pollutants near ground (°C)
    "relative_humidity_2m",      # Humidity drives particle formation (%)
    "precipitation",             # precipitation physically washes pollutants out (mm)
    "wind_speed_10m",            # Higher wind disperses pollutants (km/h)
    "wind_direction_10m",        # Direction determines pollution source area (°)
    "surface_pressure",          # Low pressure = poor air circulation (hPa)
    "weather_code",              # Affects ozone-forming photochemical reactions
]

AIR_QUALITY_VARIABLES = [
    "pm2_5",                # Fine particles — primary AQI driver (µg/m³)
    "pm10",                 # Coarse particles (µg/m³)
    "nitrogen_dioxide",     # NO₂ — traffic and industrial emissions (µg/m³)
    "sulphur_dioxide",      # SO₂ — industrial and fuel combustion (µg/m³)
    "ozone",                # O₃ — formed by photochemical reactions (µg/m³)
    "carbon_monoxide",      # CO — combustion emissions (µg/m³)
    "us_aqi",               # US AQI index — our prediction target
]


# COLUMN RENAMING
COLUMN_MAPPING = {

    # Weather Features
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "precipitation": "rain",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "surface_pressure": "pressure",
    "weather_code": "weather_code",

    # Air Quality Features
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "carbon_monoxide": "co",

    # Target Variable
    "us_aqi": "aqi",
}


# DATE HELPERS
# Helps to get end date (18-May - today's date)
def get_end_date():
    """
    Return yesterday's date as YYYY-MM-DD string.
    Archive API only has complete data up to yesterday.
    """
    return (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")


# Helps to get start date (20-Nov - past 6 month date)
def get_start_date(end_date):
    """Return Jan 1 2025 as fixed start date."""
    return "2025-01-01"


# DATA FETCH
def fetch_data(url, params, label):
    """
    Generic API fetch function.
    """

    print("\n" + "=" * 60)
    print(f"FETCHING {label.upper()} DATA")
    print("=" * 60)

    try:
        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        count = len(data.get("hourly", {}).get("time", []))

        print(f"Fetched {count} hourly records")
        return data

    except requests.exceptions.Timeout:
        print(f"[ERROR] {label} request timed out")
        return None

    except requests.exceptions.ConnectionError:
        print("[ERROR] Internet connection failed")
        return None

    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP Error: {e}")
        return None

    except Exception as e:
        print(f"[ERROR] {label} fetch failed")
        print(e)

        return None


# WEATHER DATA FETCH
def fetch_weather_data(start_date, end_date):
    """
    Fetch historical weather data.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_VARIABLES,
        "timezone": TIMEZONE,
    }

    return fetch_data(WEATHER_API_URL, params, "Weather")


# AIR QUALITY DATA FETCH
def fetch_air_quality_data(start_date, end_date):
    """
    Fetch historical air quality data.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_VARIABLES,
        "timezone": TIMEZONE,
    }

    return fetch_data(AIR_QUALITY_API_URL,params,"Air Quality",)


# MERGE DATA
def merge_data(raw_weather, raw_air_quality):
    """
    Merge weather + air quality data.
    """

    print("\n" + "=" * 60)
    print("MERGING DATA")
    print("=" * 60)

    try:

        weather_df = pd.DataFrame(raw_weather["hourly"])
        air_quality_df = pd.DataFrame(raw_air_quality["hourly"])

        # Merge datasets
        merged_df = pd.merge(weather_df, air_quality_df, on="time")

        # Rename columns
        merged_df.rename(columns=COLUMN_MAPPING, inplace=True)

        print(f"Merged shape: {merged_df.shape}")

        # Convert back to dict for JSON saving
        return merged_df.to_dict(orient="list")

    except Exception as e:
        print("[ERROR] Merge failed")
        print(e)
        return None


# SAVE DATA
def save_data(merged_dict):
    """
    Save merged data as JSON.
    """

    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)

    if merged_dict is None:
        print("  Nothing to save — merged data is None.")
        return False
    
    try:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        with open(MERGED_JSON_PATH, "w") as f:
            json.dump(merged_dict, f, indent=2)

        print(f"Saved file: {MERGED_JSON_PATH}")
        return True

    except Exception as e:
        print("[ERROR] Saving failed")
        print(e)
        return False


# MAIN PIPELINE
def run_pipeline():
    """
    Complete AQI data pipeline.
    """

    print("\n" + "=" * 60)
    print("HYDERABAD AQI DATA PIPELINE")
    print("=" * 60)

    end_date = get_end_date()

    start_date = get_start_date(end_date)

    print(f"Start Date : {start_date}")
    print(f"End Date   : {end_date}")

    # Step 1: Fetch weather data
    raw_weather = fetch_weather_data(start_date, end_date)

    
    # Step 2: Fetch AQI data
    raw_air_quality = fetch_air_quality_data(start_date,end_date)

    if raw_weather is None or raw_air_quality is None:
        print("\n[FAILED] One or both API calls failed. Aborting.")
        return False


    # Step 3: Merge datasets
    merged_dict = merge_data(raw_weather, raw_air_quality)

    if merged_dict is None:
        print("\n[FAILED] Merge failed. Aborting.")
        return False

    # Step 4: Save JSON
    saved = save_data(merged_dict)

    if saved:
        print("\nPipeline completed successfully")

    else:
        print("\nPipeline failed")

    return saved


if __name__ == "__main__":
    run_pipeline()