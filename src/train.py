# src/train.py
import argparse
import json
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import joblib
import os

FEATURE_ORDER = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

def build_model(kind: str, seed: int):
    if kind == "linear":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        meta_type = "LinearRegression"
    elif kind == "ridge":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed))])
        meta_type = "Ridge"
    elif kind == "rf":
        pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=100, random_state=seed))])
        meta_type = "RandomForest"
    else:
        raise ValueError("Unknown model kind")
    return pipe, meta_type

def main(args):
    np.random.seed(args.seed)
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    model, model_type = build_model(args.model, args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)

    # optional binary flag evaluation: top-10% as "high-risk"
    threshold = float(np.percentile(y_train, 90))
    y_bin = (y_val >= threshold).astype(int)
    pred_bin = (preds >= threshold).astype(int)
    try:
        precision = float(precision_score(y_bin, pred_bin, zero_division=0))
        recall = float(recall_score(y_bin, pred_bin, zero_division=0))
    except Exception:
        precision, recall = None, None

    metadata = {
        "model_version": args.version,
        "model_type": model_type,
        "random_seed": args.seed,
        "model_kind": args.model,
        "rmse": float(rmse),
    }

    out_dir = os.path.join(args.out_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(model, model_path, compress=3)

    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    metrics = {"rmse": float(rmse), "threshold_top10_percent": threshold, "precision_at_top10": precision, "recall_at_top10": recall}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", model_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["linear","ridge","rf"], default="linear")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", type=str, default="v0.1")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out-dir", type=str, default="models")
    args = p.parse_args()
    main(args)
