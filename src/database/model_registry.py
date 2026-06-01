"""
src/db/model_registry.py
=========================
Save and load models from MongoDB GridFS.
training_pipeline.py  → uses save_model()
dashboard.py          → uses load_model()
live_pipeline.py      → uses load_model()
"""

import pickle
import gridfs
from datetime import datetime, timezone
from config.settings import MONGO_MODELS_COLLECTION


def save_model(db, target, bundle, metrics, model_name):
    """Save model to GridFS, metadata to models collection."""
    fs         = gridfs.GridFS(db)
    models_col = db[MONGO_MODELS_COLLECTION]
    saved_at   = datetime.now(timezone.utc)

    # Delete old model for this target if exists
    old = models_col.find_one({"target": target})
    if old and "gridfs_id" in old:
        try:
            fs.delete(old["gridfs_id"])
        except Exception:
            pass

    #save new file to GridFs
    filename  = f"{target}_{model_name}_{saved_at.strftime('%Y%m%d_%H%M%S')}.pkl"
    gridfs_id = fs.put(pickle.dumps(bundle), filename=filename)

    models_col.update_one(
        {"target": target},
        {"$set": {
            "target":     target,
            "model_name": model_name,
            "gridfs_id":  gridfs_id,
            "filename":   filename,
            "saved_at":   saved_at,
            "metrics":    metrics,
            "n_features": len(bundle["feature_columns"]),
        }},
        upsert=True,
    )
    print(f"        Saved → GridFS filename: {filename}")


def load_model(db, target):
    """
    Load model bundle for given target from GridFS.
    Returns dict with keys:
        model, scaler, model_name, feature_columns, target
    """
    fs         = gridfs.GridFS(db)
    models_col = db[MONGO_MODELS_COLLECTION]

    meta = models_col.find_one({"target": target})
    if not meta:
        raise ValueError(f"No model found for target '{target}'")

    grid_out = fs.get(meta["gridfs_id"])
    bundle   = pickle.loads(grid_out.read())
    return bundle