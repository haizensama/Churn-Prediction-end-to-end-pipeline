#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# Load processed data
# -------------------------
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# -------------------------
# Set MLflow experiment
# -------------------------
mlflow.set_experiment("Churn_Prediction")

# -------------------------
# Define models
# -------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# -------------------------
# Train, log, and save models
# -------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_name", name)  # ✅ Tag the model

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # probability for positive class

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, name)

        # Save model locally for DVC tracking
        joblib.dump(model, f"models/{name}.pkl")

        # Print results
        print(f"{name} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("-" * 50)

print("✅ Training complete. Models saved in 'models/' and logged to MLflow.")
