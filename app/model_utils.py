# app/model_utils.py
import joblib
import json
import os

FEATURE_ORDER = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

def load_model(path: str = "v0.1"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Build/train model first.")
    model = joblib.load(path)
    metadata = {}
    # try to read metadata next to model
    meta_path = os.path.splitext(path)[0] + "_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        # maybe embedded attribute
        if hasattr(model, "metadata_"):
            metadata = getattr(model, "metadata_")
    return model, metadata
