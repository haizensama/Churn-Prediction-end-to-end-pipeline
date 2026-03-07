#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib

def train_model():
    # -------------------------
    # Authentication for DagsHub
    # -------------------------
    os.environ["MLFLOW_TRACKING_USERNAME"] = "haizensama"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "36c35c7a76fa8d3ab1be9fd7b1b0df25c5713e41"   # <-- replace with your token

    # -------------------------
    # MLflow tracking (DagsHub)
    # -------------------------
    MLFLOW_TRACKING_URI = "https://dagshub.com/haizensama/mlops22ug2-0179.mlflow"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = "Churn_Prediction"
    mlflow.set_experiment(experiment_name)

    # -------------------------
    # Load processed data
    # -------------------------
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # -------------------------
    # Define models + hyperparameters
    # -------------------------
    model_configs = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1], "solver": ["liblinear"]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100], "max_depth": [10, None], "min_samples_split": [2]}
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric="logloss", random_state=42),
            "params": {"n_estimators": [100], "max_depth": [3, 6], "learning_rate": [0.1]}
        }
    }

    best_model_name = None
    best_roc_auc = -1
    os.makedirs("models", exist_ok=True)

    # -------------------------
    # Train & log each model
    # -------------------------
    for name, config in model_configs.items():
        model = config["model"]
        param_grid = config["params"]

        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(X_test)

            # -------------------------
            # Metrics
            # -------------------------
            roc_auc = roc_auc_score(y_test, y_proba)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("roc_auc", roc_auc)

            # -------------------------
            # Log model artifact to MLflow (DagsHub)
            # -------------------------
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=name  # saves in run/artifacts/<model_name>
            )

            # -------------------------
            # Save pickle locally
            # -------------------------
            local_path = os.path.join("models", f"{name}.pkl")
            joblib.dump(best_model, local_path)
            print(f"{name} | ROC-AUC={roc_auc:.4f} | Saved local model: {local_path}")

            # -------------------------
            # Track best model
            # -------------------------
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = name
                mlflow.set_tag("best_model", name)

    print(f"✅ Training complete. Best model: {best_model_name}")

if __name__ == "__main__":
    train_model()
