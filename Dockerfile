# builder
FROM python:3.11-slim AS builder
WORKDIR /wheels
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir -r /wheels/requirements.txt

COPY app app
COPY models models
COPY ci/sample_payload.json ci/sample_payload.json

EXPOSE 8080
ENV MODEL_PATH=/app/models/${MODEL_VERSION}/model.joblib
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
