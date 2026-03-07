#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Authentication for DagsHub
# -----------------------------
os.environ["MLFLOW_TRACKING_USERNAME"] = "haizensama"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "36c35c7a76fa8d3ab1be9fd7b1b0df25c5713e41"   # <-- replace with your token

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/haizensama/mlops22ug2-0179.mlflow"
EXPERIMENT_NAME = "Churn_Prediction"
MODELS_FOLDER = "models"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

app = FastAPI(title="Churn Prediction API")


# -----------------------------
# Request schema
# -----------------------------
class InputData(BaseModel):
    data: list  # list of dicts, e.g., [{"tenure": 5, "MonthlyCharges": 70, ...}]


# -----------------------------
# Utility: Get best model path
# -----------------------------
def get_best_model_path():
    try:
        # 1️⃣ Check MLflow Production stage (model registry)
        registered_models = client.search_registered_models()
        for rm in registered_models:
            for v in rm.latest_versions:
                if v.current_stage == "Production":
                    best_model_name = rm.name
                    model_path = os.path.join(MODELS_FOLDER, f"{best_model_name}.pkl")
                    if os.path.exists(model_path):
                        print(f"✅ Loading Production model: {best_model_name}")
                        return model_path
                    else:
                        print(f"⚠️ Production model '{best_model_name}' found in registry but local file missing: {model_path}")

        # 2️⃣ Fallback: use best_model tag from latest runs (if any)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"❌ Experiment '{EXPERIMENT_NAME}' not found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string='tags.best_model != ""',
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if runs:
            best_model_name = runs[0].data.tags.get("best_model")
            model_path = os.path.join(MODELS_FOLDER, f"{best_model_name}.pkl")
            if os.path.exists(model_path):
                print(f"✅ Loading model from best_model tag: {best_model_name}")
                return model_path
            else:
                print(f"⚠️ Best model '{best_model_name}' found but local file missing: {model_path}")

        # 3️⃣ No model found via MLflow – look for any .pkl in models folder
        print("🔍 No model found via MLflow, checking local models folder...")
        if os.path.exists(MODELS_FOLDER):
            pkl_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pkl')]
            if pkl_files:
                # Pick the most recent? For simplicity, pick first alphabetically
                model_path = os.path.join(MODELS_FOLDER, pkl_files[0])
                print(f"✅ Using local fallback model: {pkl_files[0]}")
                return model_path

        raise ValueError("❌ No suitable model found locally or in MLflow registry.")

    except Exception as e:
        print("⚠️ Error determining best model:", e)
        return None


# -----------------------------
# Load the model on startup
# -----------------------------
best_model_path = get_best_model_path()
if best_model_path:
    model = load(best_model_path)
    print(f"🚀 Model loaded from: {best_model_path}")
else:
    model = None
    print("⚠️ API started without a valid model!")


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(input_data: InputData):
    if model is None:
        return {"error": "No model loaded"}

    df = pd.DataFrame(input_data.data)
    df.fillna(0, inplace=True)  # match training preprocessing

    try:
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
        return {"predictions": preds.tolist(), "probabilities": probs.tolist() if probs is not None else None}
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
