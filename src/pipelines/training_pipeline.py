"""
Hyderabad AQI — Training Pipeline Module, trains all the models (LinearRegression, RandomForest, XGBoost) each for different horizons
"""

import copy
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config.settings import MONGO_COLLECTION, MONGO_MODELS_COLLECTION
from src.features.feature_engineering import TARGET_COLUMNS, get_features
from src.database.database_connection import get_db_client
from src.database.model_registry import save_model


RANDOM_SEED = 42
TRAIN_RATIO = 0.80

HORIZON_MAP = {
    "aqi_24h": "24h",
    "aqi_48h": "48h",
    "aqi_72h": "72h",
}


# ── Step 1 — Fetch features from MongoDB ──────────────────────

def fetch_features(col):
    """
    Fetch all engineered feature rows from MongoDB.
    Uses the full 60-column set for fetching; each horizon
    selects its own subset during training.
    """
    print("\n" + "=" * 60)
    print("STEP 1 — FETCHING FEATURES FROM MONGODB")
    print("=" * 60)

    # Fetch all 55 features + all 3 targets
    all_features = get_features("24h")   # full 60-column list
    projection   = {c: 1 for c in all_features + TARGET_COLUMNS}
    projection["timestamp"] = 1
    projection["_id"]       = 0

    docs = list(col.find({"aqi_24h": {"$ne": None}}, projection))
    df   = pd.DataFrame(docs)
    df   = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Fetched    : {len(df):,} rows")
    print(f"  Date range : {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    print(f"  Targets    : {TARGET_COLUMNS}")

    # NaN check across all 55 features
    nan_counts = df[all_features].isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n  [WARN] NaN values found — forward-filling ...")
        print(nan_cols)
        df[all_features] = df[all_features].ffill().fillna(0)
    else:
        print("  [OK] No NaN values in features")

    return df


# ── Step 2 — Chronological train/test split ───────────────────

def prepare_split(df, horizon):
    """
    Chronological 80/20 split for a specific horizon.
    Each horizon gets its own filtered feature list and its own scaler.

    Returns:
        X_train, X_test            — raw arrays for tree models
        X_train_scaled, X_test_scaled — scaled arrays for LinearRegression
        y_train, y_test            — target series
        scaler                     — fitted StandardScaler for this horizon
        feat_cols                  — feature list used (for saving in bundle)
    """
    target    = f"aqi_{horizon}"
    feat_cols = get_features(horizon)

    df        = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    split_ts  = df["timestamp"].iloc[split_idx]

    print(f"\n  Horizon : {horizon}  |  Features : {len(feat_cols)}")
    print(f"  Train   : {df['timestamp'].iloc[0]}  →  {split_ts}  ({split_idx:,} rows)")
    print(f"  Test    : {split_ts}  →  {df['timestamp'].iloc[-1]}  "
          f"({len(df) - split_idx:,} rows)")

    X_train = df.iloc[:split_idx][feat_cols].values
    X_test  = df.iloc[split_idx:][feat_cols].values
    y_train = df.iloc[:split_idx][target]
    y_test  = df.iloc[split_idx:][target]

    # Distribution shift check
    gap  = abs(y_train.mean() - y_test.mean())
    flag = "[WARN] Large shift" if gap > 15 else "[OK]  "
    print(f"  {flag}  train_mean={y_train.mean():.1f}  "
          f"test_mean={y_test.mean():.1f}  gap={gap:.1f}")

    # Scaler fitted on this horizon's feature set only
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, \
           y_train, y_test, scaler, feat_cols


# ── Step 3 — Define models per horizon ────────────────────────
def get_models(horizon):
    """
    Per-horizon hyperparameters.
    Longer horizons use shallower trees and heavier regularization
    """
    if horizon == "24h":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                min_samples_split=20,
                max_features=0.7,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            ),
        }
    elif horizon == "48h":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                max_depth=6,             # shallower — weaker signal at 48h
                min_samples_leaf=20,
                min_samples_split=40,
                max_features=0.6,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,      # slower learning
                max_depth=4,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=8,
                reg_alpha=0.5,
                reg_lambda=3.0,          # heavier regularization
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            ),
        }
    else:  # 72h
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                max_depth=4,             # shallowest — weakest signal at 72h
                min_samples_leaf=30,
                min_samples_split=60,
                max_features=0.5,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.6,
                colsample_bytree=0.6,
                min_child_weight=12,
                reg_alpha=1.0,
                reg_lambda=5.0,          # heaviest regularization
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            ),
        }


# ── Step 4 — Train and evaluate ───────────────────────────────
def _compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def train_all_models(df):
    """
    Train 3 models × 3 horizons = 9 fits total.
    Each horizon uses its own filtered features, split, and scaler.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — TRAINING  (3 models × 3 horizons = 9 fits)")
    print("=" * 60)

    results  = {}
    trained  = {}
    scalers  = {}
    feat_map = {}

    for horizon in ["24h", "48h", "72h"]:
        print(f"\n{'─' * 60}")
        print(f"  HORIZON: {horizon}")
        print(f"{'─' * 60}")

        X_train, X_test, X_train_scaled, X_test_scaled, \
            y_train, y_test, scaler, feat_cols = prepare_split(df, horizon)

        scalers[horizon]  = scaler
        feat_map[horizon] = feat_cols

        models     = get_models(horizon)
        y_tr       = y_train.values
        y_te       = y_test.values

        # Naive baseline
        naive_pred = np.full_like(y_te, y_tr.mean(), dtype=float)
        naive      = _compute_metrics(y_te, naive_pred)

        print(f"\n  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7}  "
              f"{'TrainR²':>8}  {'Overfit':>8}")
        print(f"  {'-'*62}")
        print(f"  {'Baseline (mean)':<22} "
              f"{naive['rmse']:>7.2f} {naive['mae']:>7.2f} "
              f"{naive['r2']:>7.3f}  ← must beat this")
        print(f"  {'-'*62}")

        results[horizon] = {}
        trained[horizon] = {}

        for model_name, model in models.items():
            m = copy.deepcopy(model)

            if model_name == "LinearRegression":
                m.fit(X_train_scaled, y_tr)
                tr_pred = m.predict(X_train_scaled)
                te_pred = m.predict(X_test_scaled)
            else:
                m.fit(X_train, y_tr)
                tr_pred = m.predict(X_train)
                te_pred = m.predict(X_test)

            tr_metrics = _compute_metrics(y_tr, tr_pred)
            te_metrics = _compute_metrics(y_te, te_pred)
            overfit    = tr_metrics["r2"] - te_metrics["r2"]
            flag       = " [OVERFIT?]" if overfit > 0.15 else ""

            print(f"  {model_name:<22} "
                  f"{te_metrics['rmse']:>7.2f} "
                  f"{te_metrics['mae']:>7.2f} "
                  f"{te_metrics['r2']:>7.3f}  "
                  f"{tr_metrics['r2']:>8.3f}  "
                  f"{overfit:>8.3f}{flag}")

            results[horizon][model_name] = te_metrics
            trained[horizon][model_name] = m

    return results, trained, scalers, feat_map


# ── Step 5 — Summary ──────────────────────────────────────────
def print_summary(results):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'Horizon':<10} {'RMSE':>7} {'MAE':>7} {'R²':>7}")
    print(f"  {'-'*57}")

    for horizon in ["24h", "48h", "72h"]:
        best = min(results[horizon], key=lambda m: results[horizon][m]["rmse"])
        for model_name, m in results[horizon].items():
            marker = "  ← best" if model_name == best else ""
            print(f"  {model_name:<22} {horizon:<10} "
                  f"{m['rmse']:>7.2f} {m['mae']:>7.2f} {m['r2']:>7.3f}"
                  f"{marker}")
        print()


# ── Step 6 — Save all models ──────────────────────────────────
def save_best_models(results, trained, scalers, feat_map, db):
    """
    Save all models to MongoDB GridFS.
    Best per horizon (lowest RMSE).
    """
    print("\n" + "=" * 60)
    print("STEP 4 — SAVING ALL MODELS TO MONGODB")
    print("=" * 60)

    models_col = db[MONGO_MODELS_COLLECTION]
    deleted    = models_col.delete_many({})
    print(f"\n  Cleared {deleted.deleted_count} old model documents")

    for horizon in ["24h", "48h", "72h"]:
        target    = f"aqi_{horizon}"
        feat_cols = feat_map[horizon]
        scaler    = scalers[horizon]

        best_name = min(
            results[horizon],
            key=lambda m: results[horizon][m]["rmse"]
        )

        print(f"\n  Horizon : {horizon}  |  Target : {target}  "
              f"|  Best : {best_name}  |  Features : {len(feat_cols)}")
        print(f"  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7}  Saved as")
        print(f"  {'-'*65}")

        for model_name, model in trained[horizon].items():
            metrics = results[horizon][model_name]
            is_best = (model_name == best_name)

            bundle = {
                "model":           model,
                "scaler":          scaler,        # per-horizon scaler
                "model_name":      model_name,
                "target":          target,
                "feature_columns": feat_cols,     # filtered per horizon
                "is_best":         is_best,
            }

            save_model(db, bundle, metrics, model_name, is_best=is_best)

            star = " ★" if is_best else "  "
            print(f"  {model_name:<22}{star} "
                  f"{metrics['rmse']:>7.2f} "
                  f"{metrics['mae']:>7.2f} "
                  f"{metrics['r2']:>7.3f}")

    print("\n  All models saved successfully.")


# Main
def run_training():
    print("\n" + "=" * 60)
    print("HYDERABAD AQI — TRAINING PIPELINE")
    print("=" * 60)
    print("  Models   : LinearRegression, RandomForest, XGBoost")
    print("  Horizons : 24h (Day1), 48h (Day2), 72h (Day3)")
    print("  Features : 60 / 57 / 55  (filtered per horizon)")
    print("  Split    : 80% train / 20% test  (chronological)")
    print("=" * 60)

    client, db = get_db_client()
    col        = db[MONGO_COLLECTION]

    df = fetch_features(col)

    results, trained, scalers, feat_map = train_all_models(df)

    print_summary(results)
    save_best_models(results, trained, scalers, feat_map, db)

    client.close()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_training()