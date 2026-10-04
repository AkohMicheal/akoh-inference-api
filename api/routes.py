import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas import (
    HeartDiseaseInput,
    HeartDiseasePredictionResponse,
    PredictiveMaintenanceInput,
    PredictiveMaintenanceResponse,
)

router = APIRouter(prefix="/predict", tags=["Predictions"])

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
HEART_MODEL_PATH = os.path.join(MODELS_DIR, "heart_disease_model.pkl")
MAINTENANCE_MODEL_PATH = os.path.join(MODELS_DIR, "predictive_maintenance_model.pkl")


@router.post(
    "/heart-disease",
    response_model=HeartDiseasePredictionResponse,
    summary="Predict heart disease risk",
    description="Analyzes clinical attributes against the UCI Heart Disease model."
)
async def predict_heart_disease(payload: HeartDiseaseInput):
    """
    Receives clinical attributes and returns diagnosis risk prediction.
    Falls back to high-confidence mock output if .pkl model is pending upload.
    """
    data_dict = payload.model_dump()
    
    # If a trained model binary is present in /models, use it directly
    if os.path.exists(HEART_MODEL_PATH):
        try:
            model = joblib.load(HEART_MODEL_PATH)
            df = pd.DataFrame([data_dict])
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.85
            return HeartDiseasePredictionResponse(
                prediction=pred,
                probability=round(prob, 4),
                status="Inference completed via live model binary."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Deterministic heuristic baseline while awaiting Google Colab artifact
    # e.g., elevated chest pain type + ST depression > 2.0 flags potential risk
    is_high_risk = data_dict["oldpeak"] >= 2.0 or data_dict["cp"] > 1 or data_dict["ca"] > 0
    pred = 1 if is_high_risk else 0
    prob = 0.87 if is_high_risk else 0.15

    return HeartDiseasePredictionResponse(
        prediction=pred,
        probability=prob,
        status="Model binary pending in /models. Output generated via clinical baseline heuristic."
    )


@router.post(
    "/maintenance",
    response_model=PredictiveMaintenanceResponse,
    summary="Predict industrial equipment failure",
    description="Analyzes telemetry sensor data from AI4I predictive maintenance dataset."
)
async def predict_maintenance(payload: PredictiveMaintenanceInput):
    """
    Evaluates industrial telemetry metrics (RPM, torque, tool wear, temps)
    to forecast machine failure probability.
    """
    data_dict = payload.model_dump()

    # If a trained model binary is present in /models, use it directly
    if os.path.exists(MAINTENANCE_MODEL_PATH):
        try:
            model = joblib.load(MAINTENANCE_MODEL_PATH)
            df = pd.DataFrame([data_dict])
            pred = int(model.predict(df)[0])
            prob = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.91
            risk = "Critical" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
            return PredictiveMaintenanceResponse(
                prediction=pred,
                failure_risk_probability=round(prob, 4),
                risk_level=risk,
                status="Inference completed via live model binary."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Telemetry baseline heuristic: tool wear > 200 mins or extreme torque indicates wear
    is_critical = data_dict["tool_wear"] > 200 or data_dict["torque"] > 60.0
    pred = 1 if is_critical else 0
    prob = 0.82 if is_critical else 0.08
    risk = "Critical" if prob >= 0.7 else "Low"

    return PredictiveMaintenanceResponse(
        prediction=pred,
        failure_risk_probability=prob,
        risk_level=risk,
        status="Model binary pending in /models. Output generated via telemetry baseline heuristic."
    )
