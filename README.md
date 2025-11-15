# Random Forest Speed Prediction API

A lightweight FastAPI service that serves a trained Random Forest model to predict road segment speed. This repository contains the API entrypoint, model artifact path, and preprocessing utilities used to prepare inputs for the model.

---

**Project snapshot**

- **Purpose**: Provide predicted average speed values for map junctions / intersections using a Random Forest model.
- **Main app**: `main.py` (FastAPI)
- **Model**: `data/rf_speed_model.pkl` (loaded with `joblib`)
- **Preprocessing**: `data/preprocessing_speed.py` (contains `preprocess_data_speed` and `ALL_INTERSECTION_NAMES`)

---

**Repository structure**

- `main.py` - FastAPI application exposing `/` and `/predict/` endpoints.
- `requirements.txt` - pinned dependencies used by the project.
- `build.sh` - simple script to upgrade pip and install requirements.
- `data/` - data utilities and model artifact:
  - `preprocessing_speed.py` - preprocessing helper(s) and constants.
  - `rf_speed_model.pkl` - serialized random forest model (expected path).

---

**Quickstart (development)**

Prerequisites: Python 3.10+ (use the version that matches your environment and the wheels in `requirements.txt`). Recommended to use a virtual environment.

1. Create and activate a virtual environment (bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

2. Install dependencies (you can use the included `build.sh` on bash):

```bash
# Option A: run the build script (recommended for consistent binary wheels)
./build.sh

# Option B: direct install
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3. Run the API server locally with `uvicorn`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Health check:

```bash
curl http://127.0.0.1:8000/
```

---

**API: /predict/**

Endpoint: `POST /predict/`

Request payload (JSON) follows the Pydantic model in `main.py`:

```json
{
  "model": "randomforest",
  "coordinates": { "lat": 12.9716, "lng": 77.5946 },
  "predictionTime": "Next Hour",
  "event": null
}
```

Example `curl` request (replace coordinates as needed):

```bash
curl -s -X POST "http://127.0.0.1:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"model":"randomforest","coordinates":{"lat":12.9716,"lng":77.5946},"predictionTime":"Next Hour"}'
```

Response format (example):

```json
{
  "predictions": {
    "congestion": { "level": 0.0, "label": "Unknown" },
    "avgSpeed": 45.5
  },
  "alternativeRoute": null
}
```

Notes:
- `avgSpeed` is the number used by the consuming frontend to display predicted speed.
- The current `main.py` implementation maps coordinates to a `JunctionName` using a hardcoded placeholder; see Developer Notes below.

---

**Developer notes & TODOs**

- Coordinate mapping: `main.py` currently hardcodes `'Intersection_Trinity Circle'` for `JunctionName`. You should replace the placeholder mapping with a geospatial nearest-neighbor lookup that maps `(lat, lng)` to one of the known junction names exported by `data/preprocessing_speed.py` (e.g., `ALL_INTERSECTION_NAMES`). Consider using `scipy.spatial.cKDTree` or `geopy.distance` for this.

- Model artifact: Ensure `data/rf_speed_model.pkl` exists and matches the preprocessing pipeline of `preprocess_data_speed`. If the model was trained with a specific set of feature columns, the runtime preprocessing must produce the same feature set (order and names) or you'll receive a `feature_names mismatch` error from scikit-learn.

- Logging: `main.py` uses `logging` at INFO level. Examine logs for detailed error messages when predictions fail.

- Exception handling: `main.py` differentiates between preprocessing `ValueError` and other exceptions; expand this as needed for better error codes in the API.

---

**Troubleshooting**

- Model fails to load:
  - Confirm `data/rf_speed_model.pkl` exists and is a valid joblib pickle.
  - Make sure the Python environment uses compatible scikit-learn and joblib versions (see `requirements.txt`).

- Feature mismatch or shape errors during `predict`:
  - Validate `preprocess_data_speed` returns the same columns as used when training the model.
  - Print `processed_df.columns.tolist()` to inspect column names; `main.py` already logs this.

- Dependency issues on Windows:
  - If installation of `numpy`/`scipy` or other binary packages fails, try installing prebuilt wheels or use the `--prefer-binary` option (the included `build.sh` does this).

---

**Testing tips**

- Add unit tests for `preprocess_data_speed` that check expected columns for a variety of sample `JunctionName` inputs and datetimes.
- Add integration tests that start a test FastAPI client and POST to `/predict/` using `fastapi.testclient`.

---

**Deployment**

- For production, consider running the app with a process manager (gunicorn + uvicorn workers) and behind a reverse-proxy.
  Example (gunicorn + uvicorn workers):

```bash
gunicorn -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:8000 -w 4
```

- Ensure the model file is available in the deployment image or volume.

---

**Next steps / improvements**

- Implement geospatial nearest-neighbour mapping from coordinates -> `JunctionName`.
- Add model versioning and an API field to request/inspect model metadata.
- Add CI checks and tests, and optionally a tiny OpenAPI-based frontend or Swagger examples.

---


