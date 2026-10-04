# Model Artifacts Directory

Drop your pre-trained scikit-learn / XGBoost pipeline binaries (`.pkl`) into this directory.

### Expected Files:
1. `heart_disease_model.pkl`: Trained on UCI Heart Disease dataset.
   - Expects DataFrame columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.
2. `predictive_maintenance_model.pkl`: Trained on AI4I 2020 Predictive Maintenance dataset.
   - Expects DataFrame columns: `product_type, air_temperature, process_temperature, rotational_speed, torque, tool_wear`.

When present, `api/routes.py` automatically detects and uses them for live predictions via `joblib.load()`.
