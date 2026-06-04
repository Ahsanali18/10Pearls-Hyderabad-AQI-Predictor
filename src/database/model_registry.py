"""
src/database/model_registry.py
================================
Save and load models from MongoDB GridFS.

Who uses this:
    training_pipeline.py → save_model()
    live_pipeline.py     → load_model()   (unchanged — still works)
    dashboard/app.py     → load_model()   (unchanged — still works)

What changed vs previous version:
    save_model() now accepts is_best flag.
    Models are stored with a composite key (target + model_name),
    so all 9 trained models are kept, not just the 3 winners.
    The 3 winner models are ALSO saved under the plain target key
    so live_pipeline.py continues to work with zero changes.
"""

import pickle
import gridfs
from datetime import datetime, timezone

from config.settings import MONGO_MODELS_COLLECTION


# ─────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────

def save_model(db, storage_key, bundle, metrics, model_name, is_best=False):
    """
    Save one model bundle to GridFS and upsert its metadata document.

    Args:
        db          : MongoDB database handle
        storage_key : GridFS / collection key.
                      Use  "aqi_24h_XGBoost"      for the full comparison record.
                      Use  "aqi_24h"               for the best-model shortcut
                                                   (called a second time by training
                                                    pipeline for the winner).
        bundle      : dict — {model, scaler, model_name, target, feature_columns}
        metrics     : dict — {rmse, mae, r2}
        model_name  : str  — "LinearRegression" | "RandomForest" | "XGBoost"
        is_best     : bool — True only for the winner of each target
    """
    fs         = gridfs.GridFS(db)
    models_col = db[MONGO_MODELS_COLLECTION]
    saved_at   = datetime.now(timezone.utc)

    # ── Delete old GridFS file for this storage_key ──────────
    old = models_col.find_one({"storage_key": storage_key})
    if old and "gridfs_id" in old:
        try:
            fs.delete(old["gridfs_id"])
        except Exception:
            pass

    # ── Save new bundle to GridFS ─────────────────────────────
    filename  = f"{storage_key}_{saved_at.strftime('%Y%m%d_%H%M%S')}.pkl"
    gridfs_id = fs.put(pickle.dumps(bundle), filename=filename)

    # ── Upsert metadata document ──────────────────────────────
    models_col.update_one(
        {"storage_key": storage_key},          # unique per target+model combo
        {"$set": {
            "storage_key": storage_key,        # e.g. "aqi_24h_XGBoost"
            "target":      bundle["target"],   # e.g. "aqi_24h"
            "model_name":  model_name,         # e.g. "XGBoost"
            "is_best":     is_best,
            "gridfs_id":   gridfs_id,
            "filename":    filename,
            "saved_at":    saved_at,
            "metrics":     metrics,            # {rmse, mae, r2}
            "n_features":  len(bundle["feature_columns"]),
        }},
        upsert=True,
    )

    star = " ★ BEST" if is_best else ""
    print(f"        Saved [{storage_key}]{star}  →  GridFS: {filename}")


# ─────────────────────────────────────────────────────────
# LOAD  (unchanged interface — live_pipeline.py safe)
# ─────────────────────────────────────────────────────────

def load_model(db, target):
    """
    Load the best model bundle for a given target.

    Called by live_pipeline.py and dashboard as:
        load_model(db, "aqi_24h")
        load_model(db, "aqi_48h")
        load_model(db, "aqi_72h")

    The training pipeline saves the winner under the plain target key,
    so this lookup always returns the best model. No change needed here.

    Returns:
        dict — {model, scaler, model_name, target, feature_columns}
    """
    fs         = gridfs.GridFS(db)
    models_col = db[MONGO_MODELS_COLLECTION]

    # Look up by plain target key (e.g. "aqi_24h")
    meta = models_col.find_one({"storage_key": target})
   
    if not meta:
        raise ValueError(
            f"No model found for target '{target}'. "
            f"Run training_pipeline.py first."
        )

    grid_out = fs.get(meta["gridfs_id"])
    bundle   = pickle.loads(grid_out.read())
    return bundle