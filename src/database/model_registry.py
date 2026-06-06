import pickle
import gridfs
from datetime import datetime, timezone

from config.settings import MONGO_MODELS_COLLECTION


# SAVE
def save_model(db, bundle, metrics, model_name, is_best=False):
    """
    Save one model bundle to GridFS and upsert its metadata document.
    """
    fs         = gridfs.GridFS(db)
    models_col = db[MONGO_MODELS_COLLECTION]
    saved_at   = datetime.now(timezone.utc)
    target     = bundle["target"]

    # Delete old GridFS file for this target+model if exists
    old = models_col.find_one({"target": target, "model_name": model_name})
    if old and "gridfs_id" in old:
        try:
            fs.delete(old["gridfs_id"])
        except Exception:
            pass

    # ── Save new bundle to GridFS ─────────────────────────────
    filename  = f"{target}_{model_name}_{saved_at.strftime('%Y%m%d_%H%M%S')}.pkl"
    gridfs_id = fs.put(pickle.dumps(bundle), filename=filename)

    # ── Upsert metadata document ──────────────────────────────
    models_col.update_one(
        {"target": target, "model_name": model_name},
        {"$set": {
            "target":      target,             # e.g. "aqi_24h"
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
    print(f"        Saved [{target}_{model_name}]{star}  →  GridFS: {filename}")


# LOAD
def load_model(db, target):
    """
    Load the best model bundle for a given target.
    """
    fs         = gridfs.GridFS(db)
    models_col = db[MONGO_MODELS_COLLECTION]

    meta = models_col.find_one({"target": target, "is_best": True})
   
    if not meta:
        raise ValueError(
            f"No model found for target '{target}'. "
            f"Run training_pipeline.py first."
        )

    grid_out = fs.get(meta["gridfs_id"])
    bundle   = pickle.loads(grid_out.read())
    return bundle