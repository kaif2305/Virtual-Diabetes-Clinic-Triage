# app/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model_utils import FEATURE_ORDER, load_model
import os

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v0.1")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", MODEL_VERSION, "model.joblib")
MODEL_PATH = os.path.normpath(MODEL_PATH)

app = FastAPI(title="Virtual Diabetes Clinic Triage")

class Features(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

model, metadata = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata.get("model_version", "unknown"), "model_type": metadata.get("model_type", "unknown")}

@app.post("/predict")
def predict(feat: Features):
    try:
        x = [getattr(feat, f) for f in FEATURE_ORDER]
        pred = model.predict([x])[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
