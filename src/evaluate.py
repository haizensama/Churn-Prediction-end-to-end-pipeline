#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# -------------------------
# Load test data
# -------------------------
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# -------------------------
# Set MLflow experiment
# -------------------------
mlflow.set_experiment("Churn_Prediction")

# -------------------------
# Ensure evaluation folder exists
# -------------------------
os.makedirs("evaluation", exist_ok=True)

# -------------------------
# List of models to evaluate
# -------------------------
models = ["LogisticRegression", "RandomForest", "XGBoost"]

for model_name in models:

    print(f"Evaluating {model_name}...")
    model = joblib.load(f"models/{model_name}.pkl")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # -------------------------
    # Metrics
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # -------------------------
    # Confusion Matrix
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_path = f"evaluation/{model_name}_confusion_matrix.png"
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()

    # -------------------------
    # ROC Curve
    # -------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_curve = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_curve:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right")

    roc_path = f"evaluation/{model_name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # -------------------------
    # Log to MLflow
    # -------------------------
    with mlflow.start_run(run_name=f"{model_name}_evaluation"):

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

    print(f"{model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("-" * 50)

print("✅ Evaluation complete. Metrics and artifacts logged to MLflow.")
