# CHANGELOG

All notable changes to the Virtual Diabetes Clinic Triage system.

---

## [v0.2] 

### Added
- Ridge and RandomForestRegressor options.
- High-risk flag (top-10% threshold).
- Metrics logged: RMSE, precision, recall.
- MODEL_VERSION support in API for multiple versions.

### Changed
- Production model: Ridge (better RMSE than LinearRegression).
- Training script accepts `--model` argument.
- Docker/CI updated to handle versioned models.

### Fixed
- MODEL_PATH issues for multiple versions.
- Metadata fallback if file missing.

### Metrics Comparison

**Regression Performance**

| Metric | v0.1 | v0.2 | Delta |
|--------|------|------|-------|
| RMSE   | 53.853 | 52.740 | -1.113 |

**High-Risk Flag (top-10%)**

| Metric    | Value |
|-----------|-------|
| Precision | 0.72  |
| Recall    | 0.56  |
| Threshold | 271.4 |
| Flagged   | 44/442 patients |

---

## [v0.1] 

### Added
- LinearRegression + StandardScaler baseline.
- REST API `/health` and `/predict`.
- Docker container.
- CI/CD workflow and tests.

### Metrics
- RMSE: 53.853
- Training time: ~0.5s
- Model size: ~2KB
- API response: <100ms
