# tests/test_training.py
import subprocess
import os
import json
import tempfile

def test_train_creates_files():
    out = tempfile.mkdtemp()
    cmd = ["python", "src/train.py", "--model", "linear", "--seed", "0", "--version", "test", "--out-dir", out]
    subprocess.run(cmd, check=True)
    model_path = os.path.join(out, "test", "model.joblib")
    metrics_path = os.path.join(out, "test", "metrics.json")
    assert os.path.exists(model_path)
    assert os.path.exists(metrics_path)
    with open(metrics_path, "r") as f:
        m = json.load(f)
    assert "rmse" in m
