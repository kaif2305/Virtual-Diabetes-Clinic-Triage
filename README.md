# 🏥 Virtual Diabetes Clinic Triage API: (MLOps assignment)

This assignment explains the architecture and components of a minimal Machine Learning Operations (MLOps) project designed to predict diabetes-related risk. It uses a **FastAPI** service to deploy a model trained on the standard scikit-learn diabetes dataset, focusing on reproducibility and a robust CI/CD pipeline for automated containerized deployment.

---

## 🚀 Overview and Purpose

The project's primary goal is to provide a simple, containerized, and fully automated mechanism for serving a pre-trained predictive model. By leveraging **FastAPI**, **Docker**, and **GitHub Actions**, it demonstrates a complete, end-to-end MLOps workflow, from development to production-ready image publishing on the **GitHub Container Registry (GHCR)**.  

The model acts as a basic triage tool, predicting a quantitative diabetes disease progression metric based on ten patient features.

---

## 📂 Repository Structure

The file structure is organized to separate application logic, training code, and CI/CD configuration clearly:

| Path | Purpose |
|------|---------|
| `.github/workflows/` | Contains GitHub Actions workflows for CI (`pr.yml`) and Release (`release.yml`). |
| `app/` | Holds the FastAPI service (`app.py`) and model utility functions (`model_utils.py`). |
| `src/` | Contains the independent training script (`train.py`) for generating model artifacts. |
| `ci/` | Stores utility files, such as the `sample_payload.json` for API smoke testing. |
| `tests/` | Contains unit tests for API endpoints and the model training process. |
| `models/` | Artifact directory where trained models (`.joblib`) and metadata (`.json`) are stored. |
| `requirements.txt` | Defines all Python dependencies, essential for reproducible environments. |
| `Dockerfile` | Multi-stage build configuration for creating a slim, production-ready container image. |

---

## ⚙️ Core Python Dependencies

The project relies on a focused set of libraries, defined in `requirements.txt`:

- **API Framework:** `fastapi` and `uvicorn` power the lightweight and fast web service.
- **Machine Learning:** `scikit-learn` provides the diabetes dataset, models (Linear Regression, Ridge, Random Forest), and data preprocessing tools.
- **Serialization:** `joblib` is used to efficiently save and load the trained ML model object.
- **Validation:** `pydantic` enforces strict input schemas for the API, ensuring data quality and consistency.
- **Testing / Quality:** `pytest` (unit testing) and `flake8` (linting) enforce code quality during CI.

---

## 💻 FastAPI Service (`app/app.py`)

The service exposes the model via two standard REST endpoints:

### **1. Health Check (`GET /health`)**
- Returns a `200 OK` status.
- Provides metadata about the loaded model, including its **version** and **type**.
- Crucial for verifying service readiness and deployment integrity.

### **2. Prediction (`POST /predict`)**
- Accepts a JSON body containing the **10 required features** (`age, sex, bmi, bp, s1...s6`).
- Pydantic validates the input structure.
- Transforms the input array, passes it to the loaded model, and returns a single predicted diabetes progression metric as a float.
- A default `MODEL_PATH` (`/app/models/model.joblib`) is used but can be overridden via an environment variable for flexibility.

---

## 🧠 Model Training (`src/train.py`)

The training script is designed for reproducible artifact generation:

- **Model Options:** Supports training a `linear` (Linear Regression), `ridge`, or `rf` (Random Forest) regressor.
- **Pipeline:** Always uses a `StandardScaler` to normalize the input features before training.
- **Reproducibility:** Fixed random seed ensures identical results from repeated runs.
- **Artifacts Generated:**
  - `model.joblib`: The serialized, trained model object (including the StandardScaler).
  - `model_metadata.json`: Key model details (version, type, seed) for deployment transparency.
  - `metrics.json`: Performance metrics like RMSE on the validation set.

- **Triage Evaluation:** Optional binary "high-risk" evaluation using the top 10% of predictions as a threshold, saving precision/recall metrics.

---

## 📦 Containerization with Multi-Stage Docker

The project uses a **multi-stage Dockerfile** to minimize final image size and reduce build complexity.

### **Builder Stage**
- Installs build tools and Python dependencies from `requirements.txt`.
- Uses `pip wheel` to package dependencies into `.whl` files for deterministic builds.

### **Runtime Stage**
- Starts from a small base image (e.g., Python slim).
- Installs dependencies from the pre-built `.whl` files.
- Copies the application code, sample payload, and trained model artifacts.
- Exposes port `8080` and runs the service via `uvicorn`.

> This ensures deterministic dependency installation and a minimal attack surface by excluding unnecessary build tools from the final image.

---

## 🔄 GitHub Actions: The CI/CD Pipeline

Two main workflows automate quality control and deployment:

### **1. PR / Push CI (`pr.yml`)**
- **Trigger:** Runs on every push to `main` and on every Pull Request.
- **Purpose:** Ensures code quality and basic functionality before merging.
- **Steps:**
  - Install dependencies.
  - Run unit tests (`pytest`) and linting (`flake8`).
  - Perform a quick "smoke-train" of the model to verify the training script.

### **2. Release Workflow (`release.yml`)**
- **Trigger:** Runs only when a new Git Tag is pushed (e.g., `v1.0`).
- **Purpose:** Prepares and publishes final production artifacts.
- **Steps:**
  - Train the final production model (e.g., `ridge` by default).
  - Build the Docker image using the multi-stage Dockerfile.
  - Log in to GHCR and push the container image with the corresponding tag (e.g., `v1.0`).
  - Create a GitHub Release and upload the generated `metrics.json` as a release asset for historical performance tracking.
