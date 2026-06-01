"""
src/pipelines/backfill_pipeline.py
====================================
Hyderabad AQI — Backfill Pipeline

Loads historical merged JSON → applies feature engineering →
upserts all rows into MongoDB Atlas (Feature Store).

Re-run this whenever feature_engineering.py schema changes
(e.g. adding/removing columns).

Steps:
    1. Load  raw JSON from data/raw/aqi_merged.json
    2. Run   engineer_features() → 38 features + 3 targets
    3. Connect to MongoDB
    4. Upsert all rows in batches of 500
"""

import json
import math
import os
import sys
from datetime import timezone

import numpy as np
import pandas as pd
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from config.settings import MERGED_JSON_PATH, MONGO_DB_NAME, MONGO_COLLECTION
from src.features.feature_engineering import (
    engineer_features,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)
from src.database.database_connection import get_db_client, nan_to_none

BATCH_SIZE = 500


# HELPERS
def _row_to_doc(ts, row):
    """Convert a DataFrame row to a MongoDB document."""
    doc = {"timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc)}
    for col, val in row.items():
        if isinstance(val, np.integer):
            doc[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            doc[col] = nan_to_none(float(val))
        else:
            doc[col] = val
    return doc


def _build_upsert_ops(df):
    """Build a list of UpdateOne upsert operations from the DataFrame."""
    ops = []
    for ts, row in df.iterrows():
        doc = _row_to_doc(ts, row)
        ops.append(UpdateOne(
            {"timestamp": doc["timestamp"]},   # match on unique timestamp
            {"$set": doc},
            upsert=True,
        ))
    return ops


def _push_batches(collection, ops):
    """Execute upsert operations in batches and print progress."""
    total_upserted = 0
    total_modified = 0
    total_batches  = math.ceil(len(ops) / BATCH_SIZE)

    for i in range(0, len(ops), BATCH_SIZE):
        batch     = ops[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}  ({len(batch)} rows) ...", end=" ")
        try:
            result          = collection.bulk_write(batch, ordered=False)
            total_upserted += result.upserted_count
            total_modified += result.modified_count
            print(f"upserted={result.upserted_count}  modified={result.modified_count}")
        except BulkWriteError as bwe:
            print(f"[WARN] BulkWriteError in batch {batch_num}")
            for err in bwe.details.get("writeErrors", []):
                print(f"       {err}")

    return {"upserted": total_upserted, "modified": total_modified}


# MAIN RUNNER
def run_backfill():
    print("\n" + "=" * 60)
    print("BACKFILL PIPELINE — Historical JSON → MongoDB")
    print("=" * 60)

    # Step 1: Load raw JSON 
    print(f"\n[1/4] Loading raw data from:\n      {MERGED_JSON_PATH}")

    if not os.path.exists(MERGED_JSON_PATH):
        print(f"[ERROR] File not found: {MERGED_JSON_PATH}")
        sys.exit(1)

    with open(MERGED_JSON_PATH, "r") as f:
        raw = json.load(f)

    df_raw = pd.DataFrame(raw)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw.set_index("time", inplace=True)
    df_raw.sort_index(inplace=True)

    print(f"  Loaded  : {len(df_raw):,} raw rows")
    print(f"  Range   : {df_raw.index.min()}  →  {df_raw.index.max()}")
    print(f"  Columns : {list(df_raw.columns)}")

    # Step 2: Perform Feature engineering 
    print("\n[2/4] Running feature engineering ...")
    df_features = engineer_features(df_raw)

    print(f"\n  Output shape : {df_features.shape}")
    print(f"  Features     : {len(FEATURE_COLUMNS)}")
    print(f"  Targets      : {TARGET_COLUMNS}")

    nan_count = df_features.isnull().sum().sum()
    if nan_count > 0:
        print(f"\n  [WARN] {nan_count} NaN values remain after dropna — investigate!")
        print(df_features.isnull().sum()[df_features.isnull().sum() > 0])
    else:
        print("  [OK] Zero NaN values — data is clean")

    # Step 3: Connect to MongoDB database
    print(f"\n[3/4] Connecting to MongoDB ...")
    print(f"  DB         : {MONGO_DB_NAME}")
    print(f"  Collection : {MONGO_COLLECTION}")

    client, db = get_db_client()
    collection = db[MONGO_COLLECTION]

    # Ensure unique index on timestamp
    collection.create_index("timestamp", unique=True, background=True)
    print("  [OK] Connected. Unique index on 'timestamp' ensured.")

    # Step 4: Upsert in batches
    print(f"\n[4/4] Upserting {len(df_features):,} rows in batches of {BATCH_SIZE} ...")
    ops    = _build_upsert_ops(df_features)
    counts = _push_batches(collection, ops)

    # Final collection count
    total_in_db = collection.count_documents({})
    client.close()

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print(f"  Rows processed    : {len(df_features):,}")
    print(f"  New inserts       : {counts['upserted']:,}")
    print(f"  Updates (re-run)  : {counts['modified']:,}")
    print(f"  Total in MongoDB  : {total_in_db:,}")
    print("=" * 60)


if __name__ == "__main__":
    run_backfill()