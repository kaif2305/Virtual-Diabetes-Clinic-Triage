# 🏥 Virtual Diabetes Clinic Triage API

A minimal MLOps project to predict diabetes progression using a pre-trained ML model.  
Uses **FastAPI** for API, **Docker** for containerization, and **GitHub Actions** for CI/CD.

---

## 🚀 Overview

- Predicts a diabetes progression score from **10 patient features** (`age, sex, bmi, bp, s1..s6`).
- Supports **LinearRegression, Ridge, and RandomForest** models.
- Provides **versioned models** via `MODEL_VERSION` environment variable.
- Includes **high-risk patient flag** for top-10% scores.
- Fully containerized and CI/CD ready.

---

## 📂 Repo Structure

| Path | Purpose |
|------|---------|
| `app/` | FastAPI service (`app.py`) + utilities (`model_utils.py`) |
| `src/` | Training script (`train.py`) |
| `tests/` | Unit tests for API & training |
| `ci/` | Sample payload for testing |
| `models/` | Trained models and metrics (generated per version) |
| `.github/workflows/` | CI/CD workflows (`pr.yml`, `release.yml`) |
| `Dockerfile` | Multi-stage Docker build |
| `requirements.txt` | Python dependencies |

---

## 💻 API Endpoints

### **GET /health**
Returns `status: ok` and model metadata (`version`, `type`).

### **POST /predict**
- Input: JSON of 10 features.
- Output: `{"prediction": float}`
- Optional high-risk flag in v0.2.

---

## 🧠 Training (`src/train.py`)

- Supports `--model linear|ridge|rf`.
- Uses `StandardScaler` + model pipeline.
- Generates:
  - `model.joblib`
  - `model_metadata.json`
  - `metrics.json`
- Optional: high-risk evaluation (top-10% predictions).
- Reproducible via `--seed`.

---

## 📦 Docker

- Multi-stage build:
  1. **Builder**: installs dependencies and builds wheels.
  2. **Runtime**: copies app & model artifacts, runs `uvicorn`.
- Exposes port `8080`.

---

## 🔄 CI/CD

### **PR / Push (`pr.yml`)**
- Runs on push/PR.
- Installs deps, runs tests, lints, quick smoke-train.

### **Release (`release.yml`)**
- Runs on Git tag push.
- Trains production model (`ridge` by default), builds & pushes Docker image to GHCR.
- Creates GitHub Release with metrics.

---

## ▶️ Quick Start (Local)

```bash
# 1. Create virtual environment & install
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train baseline model v0.1
python src/train.py --model linear --seed 42 --version v0.1 --out-dir models
mkdir -p models && cp models/v0.1/model.joblib models/model.joblib
cp models/v0.1/model_metadata.json models/model_metadata.json
cp models/v0.1/metrics.json models/metrics.json

# 3. Run API locally
uvicorn app.app:app --host 0.0.0.0 --port 8080

# 4. Test endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @ci/sample_payload.json

# 5. Docker build & run
docker build -t local/vdt:v0.1 .
docker run -p 8080:8080 local/vdt:v0.1
