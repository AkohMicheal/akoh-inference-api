from pydantic import BaseModel, Field
from typing import Literal, Optional


class HeartDiseaseInput(BaseModel):
    """
    Input schema for the UCI Heart Disease Classification Model.
    All attributes correspond to clinical indicators evaluated during diagnosis.
    """
    age: int = Field(..., ge=1, le=120, description="Age of the patient in years", examples=[58])
    sex: int = Field(..., ge=0, le=1, description="Gender (1 = male, 0 = female)", examples=[1])
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", examples=[2])
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure in mm Hg", examples=[132])
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol in mg/dl", examples=[224])
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", examples=[0])
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0, 1, or 2)", examples=[0])
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved", examples=[173])
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)", examples=[0])
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest", examples=[3.2])
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0, 1, or 2)", examples=[2])
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-3) colored by flourosopy", examples=[2])
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)", examples=[3])


class HeartDiseasePredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No heart disease detected, 1 = Heart disease indicated")
    probability: float = Field(..., description="Estimated prediction confidence / probability")
    status: str = Field(..., description="Execution status or model note")


class PredictiveMaintenanceInput(BaseModel):
    """
    Input schema for the AI4I 2020 Predictive Maintenance Dataset.
    Reflects synthetic CNC milling machine operational telemetry.
    """
    product_type: Literal["L", "M", "H"] = Field(..., description="Product quality variant (L = Low, M = Medium, H = High)", examples=["M"])
    air_temperature: float = Field(..., ge=250.0, le=350.0, description="Air temperature in Kelvin", examples=[298.2])
    process_temperature: float = Field(..., ge=250.0, le=350.0, description="Process temperature in Kelvin", examples=[308.7])
    rotational_speed: int = Field(..., ge=500, le=4000, description="Rotational speed in RPM", examples=[1408])
    torque: float = Field(..., ge=0.0, le=200.0, description="Torque in Nm", examples=[46.3])
    tool_wear: int = Field(..., ge=0, le=500, description="Tool wear in minutes", examples=[182])


class PredictiveMaintenanceResponse(BaseModel):
    prediction: int = Field(..., description="0 = Normal operation, 1 = Machine failure predicted")
    failure_risk_probability: float = Field(..., description="Estimated probability of failure")
    risk_level: str = Field(..., description="Categorical risk assessment (Low, Medium, Critical)")
    status: str = Field(..., description="Execution status or model note")
