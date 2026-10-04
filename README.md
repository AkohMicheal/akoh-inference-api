# Akoh Inference API

A lightweight, containerized FastAPI microservice for serving machine learning models with strict runtime schema validation and low latency.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Overview

Most ML projects die inside Jupyter notebooks. This service turns trained scikit-learn models into reliable, production-ready HTTP endpoints that any frontend or microservice can call with confidence.

Instead of writing loose Python dictionaries and hoping the client sends the right datatypes, this service enforces strict Pydantic schemas at the gateway. Invalid payloads are rejected with descriptive 422 errors before they ever touch the model.

### Served Models
1. **Clinical Cardiology Diagnostic**: Evaluates patient clinical markers (resting blood pressure, cholesterol, ST depression, fluoroscopy vessels) to predict heart disease risk (trained on the UCI Heart Disease dataset).
2. **Industrial Predictive Maintenance**: Consumes live CNC milling machine telemetry (rotational speed, torque, air/process temperature, tool wear) to forecast equipment failure before mechanical breakdown (trained on the AI4I 2020 dataset).

---

## Architectural Decisions

- **Strict Contract Enforcement**: Pydantic models validate range bounds (e.g., body temp, RPM ranges) and categorical types (`Literal["L", "M", "H"]`) at the network boundary.
- **Dynamic Binary Loader**: The `/predict` router checks the `models/` directory for serialized `.pkl` pipelines via `joblib`. If a model binary is present, it uses it immediately. If a model is currently training or updating, it falls back to a deterministic baseline heuristic so dependent services don't crash.
- **Minimal Footprint**: Built on `python:3.11-slim` with a multi-stage-ready Dockerfile, keeping image size compact and cold-start times under 1 second.
- **Zero-Friction Documentation**: Native Swagger UI (`/docs`) and ReDoc (`/redoc`) enabled out of the box for client engineers to test payloads interactively.

---

## Project Structure

```
akoh-inference-api/
├── api/
│   ├── __init__.py
│   └── routes.py         # /predict endpoints & inference logic
├── models/
│   ├── .gitkeep
│   └── README.md         # Instructions for dropping serialized .pkl pipelines
├── Dockerfile            # Container definition (Python 3.11-slim)
├── LICENSE               # MIT
├── main.py               # FastAPI entrypoint, middleware, health check
├── requirements.txt      # Locked dependencies
├── schemas.py            # Pydantic schemas with type & range constraints
└── README.md
```

---

## Quickstart

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/AkohMicheal/akoh-inference-api.git
cd akoh-inference-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000
```

The interactive API docs will be live at `http://localhost:8000/docs`.

### 2. Run with Docker

```bash
# Build the container
docker build -t akoh-inference-api .

# Run container
docker run -p 8000:8000 akoh-inference-api
```

---

## API Endpoints

### 1. Heart Disease Prediction
- **Endpoint**: `POST /predict/heart-disease`
- **Payload**:
```json
{
  "age": 58,
  "sex": 1,
  "cp": 2,
  "trestbps": 132,
  "chol": 224,
  "fbs": 0,
  "restecg": 0,
  "thalach": 173,
  "exang": 0,
  "oldpeak": 3.2,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```
- **Response**:
```json
{
  "prediction": 1,
  "probability": 0.87,
  "status": "Inference completed via live model binary."
}
```

### 2. Predictive Maintenance
- **Endpoint**: `POST /predict/maintenance`
- **Payload**:
```json
{
  "product_type": "M",
  "air_temperature": 298.2,
  "process_temperature": 308.7,
  "rotational_speed": 1408,
  "torque": 46.3,
  "tool_wear": 182
}
```
- **Response**:
```json
{
  "prediction": 0,
  "failure_risk_probability": 0.08,
  "risk_level": "Low",
  "status": "Inference completed via live model binary."
}
```

### 3. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "models": {
    "heart_disease": "ready",
    "predictive_maintenance": "ready"
  }
}
```

---

## Author

**Micheal Akoh-Idoko**
- Portfolio: [michealakohportfolio.vercel.app](https://michealakohportfolio.vercel.app/)
- LinkedIn: [linkedin.com/in/micheal-akoh](https://linkedin.com/in/micheal-akoh)
- GitHub: [@AkohMicheal](https://github.com/AkohMicheal)

---

## License

This project is licensed under the [MIT License](LICENSE).
