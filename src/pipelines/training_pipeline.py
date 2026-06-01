"""
src/pipelines/training_pipeline.py
====================================
Hyderabad AQI — Training Pipeline

Trains 3 models per target (no hyperparameter tuning):
    1. LinearRegression  — linear baseline
    2. RandomForest      — tree ensemble, handles non-linearity well
    3. XGBoost           — gradient boosting, typically strongest performer

Targets (point-ahead):
    aqi_24h  → AQI at t+24h  (Day 1)
    aqi_48h  → AQI at t+48h  (Day 2)
    aqi_72h  → AQI at t+72h  (Day 3)

Split:
    Strict chronological 80/20 — NO shuffling.
    Time series must never be shuffled or future data leaks into training.

Evaluation metrics:
    RMSE — penalises large errors heavily (good for AQI spikes)
    MAE  — average absolute error in AQI units (easy to explain)
    R²   — fraction of variance explained (0=baseline, 1=perfect)

Saves:
    Best model per target (lowest test RMSE) → MongoDB GridFS
    Bundle includes: model + scaler + feature_columns + metadata
"""

import copy
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config.settings import MONGO_COLLECTION
from src.features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS
from src.database.database_connection import get_db_client
from src.database.model_registry import save_model

RANDOM_SEED = 42
TRAIN_RATIO = 0.80


# FETCH FEATURES FROM MONGODB
def fetch_features(col):
    """
    Fetch all engineered feature rows from MongoDB.
    Filter: only rows where aqi_24h is not null
    (ensures all 3 targets are present — backfill dropped the tail rows).
    """
    print("\n" + "=" * 60)
    print("STEP 1 — FETCHING FEATURES FROM MONGODB")
    print("=" * 60)

    projection = {c: 1 for c in FEATURE_COLUMNS + TARGET_COLUMNS}
    projection["timestamp"] = 1
    projection["_id"]       = 0

    docs = list(col.find({"aqi_24h": {"$ne": None}}, projection))
    df   = pd.DataFrame(docs)
    df   = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Fetched    : {len(df):,} rows")
    print(f"  Date range : {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")
    print(f"  Features   : {len(FEATURE_COLUMNS)}")
    print(f"  Targets    : {TARGET_COLUMNS}")

    # Sanity: check for NaN in feature columns
    nan_counts = df[FEATURE_COLUMNS].isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n  [WARN] NaN values found — forward-filling ...")
        print(nan_cols)
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill().fillna(0)
    else:
        print("  [OK] No NaN values in features")

    return df


# CHRONOLOGICAL TRAIN / TEST SPLIT
def prepare_splits(df):
    print("\n" + "=" * 60)
    print("STEP 2 — CHRONOLOGICAL TRAIN/TEST SPLIT  (80/20)")
    print("=" * 60)

    df        = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    split_ts  = df["timestamp"].iloc[split_idx]

    print(f"  Train : {df['timestamp'].iloc[0]}  →  {split_ts}  ({split_idx:,} rows)")
    print(f"  Test  : {split_ts}  →  {df['timestamp'].iloc[-1]}  "
          f"({len(df) - split_idx:,} rows)")

    X_train = df.iloc[:split_idx][FEATURE_COLUMNS].values
    X_test  = df.iloc[split_idx:][FEATURE_COLUMNS].values
    y_train = df.iloc[:split_idx][TARGET_COLUMNS]
    y_test  = df.iloc[split_idx:][TARGET_COLUMNS]

    # Check for distribution shift between train and test
    print()
    for target in TARGET_COLUMNS:
        tr_mean = y_train[target].mean()
        te_mean = y_test[target].mean()
        gap     = abs(tr_mean - te_mean)
        flag    = "[WARN] Large shift" if gap > 15 else "[OK]  "
        print(f"  {flag}  {target:<10}  "
              f"train_mean={tr_mean:.1f}  test_mean={te_mean:.1f}  "
              f"gap={gap:.1f}")

    # Scaler for Linear Regression (tree models don't need scaling)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# DEFINE MODELS
def get_models():
    """
    LinearRegression : needs_scale=True  (features must be standardised)
    RandomForest     : needs_scale=False (tree-based, scale-invariant)
    XGBoost          : needs_scale=False (tree-based, scale-invariant)
    """
    return {
        "LinearRegression": {
            "model":       LinearRegression(),
            "needs_scale": True,
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=300,        # 300 trees — good bias-variance balance
                max_depth=10,            # Prevents overfitting on 12k rows
                min_samples_leaf=10,     # Smooths leaf predictions
                max_features=0.7,        # 70% features per split — reduces correlation
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "needs_scale": False,
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=500,        # More trees needed for boosting
                max_depth=5,             # Shallow trees, many of them
                learning_rate=0.05,      # Small steps = more stable learning
                subsample=0.8,           # 80% row subsampling per tree
                colsample_bytree=0.8,    # 80% feature subsampling per tree
                min_child_weight=5,      # Regularisation — min samples per leaf
                reg_alpha=0.1,           # L1 regularisation
                reg_lambda=1.0,          # L2 regularisation
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            ),
            "needs_scale": False,
        },
    }


# TRAIN & EVALUATE
def _compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R² and return as dict."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def train_all_models(X_train, X_test, X_train_scaled, X_test_scaled,
                     y_train, y_test):
    """
    Train each of the 3 models on each of the 3 targets.
    Prints a comparison table per target.

    Returns:
        results : dict[target][model_name] = {rmse, mae, r2}
        trained : dict[target][model_name] = fitted model object
    """
    print("\n" + "=" * 60)
    print("STEP 3 — TRAINING  (3 models × 3 targets = 9 fits)")
    print("=" * 60)

    models  = get_models()
    results = {t: {} for t in TARGET_COLUMNS}
    trained = {t: {} for t in TARGET_COLUMNS}

    for target in TARGET_COLUMNS:
        y_tr = y_train[target].values
        y_te = y_test[target].values

        # Naive baseline — always predict training mean
        naive_pred = np.full_like(y_te, y_tr.mean(), dtype=float)
        naive      = _compute_metrics(y_te, naive_pred)

        print(f"\n  ── {target} ──────────────────────────────────────────")
        print(f"  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7}  "
              f"{'TrainR²':>8}  {'Overfit':>8}")
        print(f"  {'-'*62}")
        print(f"  {'Baseline (mean)':<22} "
              f"{naive['rmse']:>7.2f} {naive['mae']:>7.2f} "
              f"{naive['r2']:>7.3f}  ← must beat this")
        print(f"  {'-'*62}")

        for model_name, cfg in models.items():
            m = copy.deepcopy(cfg["model"])

            if cfg["needs_scale"]:
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
                  f"{overfit:>8.3f}"
                  f"{flag}")

            results[target][model_name] = te_metrics
            trained[target][model_name] = m

    return results, trained


# SUMMARY TABLE
def print_summary(results):
    """Print a clean final summary across all targets and models."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22} {'Target':<12} {'RMSE':>7} {'MAE':>7} {'R²':>7}")
    print(f"  {'-'*54}")

    for target in TARGET_COLUMNS:
        best = min(results[target], key=lambda m: results[target][m]["rmse"])
        for model_name, m in results[target].items():
            marker = "  ← best" if model_name == best else ""
            print(f"  {model_name:<22} {target:<12} "
                  f"{m['rmse']:>7.2f} {m['mae']:>7.2f} {m['r2']:>7.3f}"
                  f"{marker}")
        print()


# SAVE BEST MODEL PER TARGET
def save_best_models(results, trained, scaler, db):
    """
    Pick the model with lowest test RMSE per target.
    Save a bundle {model, scaler, feature_columns, metadata} to MongoDB GridFS.
    """
    print("\n" + "=" * 60)
    print("STEP 4 — SAVING BEST MODELS TO MONGODB")
    print("=" * 60)

    for target in TARGET_COLUMNS:
        best_name    = min(results[target], key=lambda m: results[target][m]["rmse"])
        best_model   = trained[target][best_name]
        best_metrics = results[target][best_name]

        print(f"\n  Target : {target}")
        print(f"  Winner : {best_name}")
        print(f"  Metrics: RMSE={best_metrics['rmse']:.2f}  "
              f"MAE={best_metrics['mae']:.2f}  "
              f"R²={best_metrics['r2']:.3f}")

        bundle = {
            "model":           best_model,
            "scaler":          scaler,
            "model_name":      best_name,
            "target":          target,
            "feature_columns": FEATURE_COLUMNS,
        }

        save_model(db, target, bundle, best_metrics, best_name)
        print(f"  Saved  → MongoDB model registry")

    print("\n  All 3 best models saved successfully.")


# MAIN
def run_training():
    print("\n" + "=" * 60)
    print("HYDERABAD AQI — TRAINING PIPELINE")
    print("=" * 60)
    print("  Models  : LinearRegression, RandomForest, XGBoost")
    print("  Targets : aqi_24h (Day1), aqi_48h (Day2), aqi_72h (Day3)")
    print("  Tuning  : OFF — sensible defaults")
    print("  Split   : 80% train / 20% test  (chronological)")
    print("=" * 60)

    client, db = get_db_client()
    col        = db[MONGO_COLLECTION]

    df = fetch_features(col)

    X_train, X_test, X_train_scaled, X_test_scaled, \
        y_train, y_test, scaler = prepare_splits(df)

    results, trained = train_all_models(
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
    )

    print_summary(results)
    save_best_models(results, trained, scaler, db)

    client.close()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_training()