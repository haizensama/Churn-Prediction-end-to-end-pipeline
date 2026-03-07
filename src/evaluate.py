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


def evaluate_model():

    # -------------------------
    # Authentication for DagsHub
    # -------------------------
    os.environ["MLFLOW_TRACKING_USERNAME"] = "haizensama"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "36c35c7a76fa8d3ab1be9fd7b1b0df25c5713e41"   # <-- replace with your token

    # -------------------------
    # MLflow tracking (DagsHub)
    # -------------------------
    mlflow.set_tracking_uri("https://dagshub.com/haizensama/mlops22ug2-0179.mlflow")

    # -------------------------
    # Set MLflow experiment
    # -------------------------
    mlflow.set_experiment("Churn_Prediction")

    # -------------------------
    # Load test data
    # -------------------------
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Handle missing values
    X_test = X_test.fillna(0)

    # -------------------------
    # Ensure evaluation folder exists
    # -------------------------
    os.makedirs("evaluation", exist_ok=True)

    # -------------------------
    # List of models
    # -------------------------
    models = ["LogisticRegression", "RandomForest", "XGBoost"]

    best_model_name = None
    best_roc_auc = 0
    best_f1 = 0

    # -------------------------
    # Evaluate each model
    # -------------------------
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

        # Track best model
        if roc_auc > best_roc_auc or (roc_auc == best_roc_auc and f1 > best_f1):
            best_model_name = model_name
            best_roc_auc = roc_auc
            best_f1 = f1

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
        # Log evaluation to MLflow
        # -------------------------
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):

            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("stage", "evaluation")

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

    # -------------------------
    # Log best model selection
    # -------------------------
    with mlflow.start_run(run_name="best_model_selection"):

        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_metric("best_model_roc_auc", best_roc_auc)
        mlflow.log_metric("best_model_f1", best_f1)

    print(f"✅ Evaluation complete. Best model: {best_model_name}")
    print("Metrics and artifacts logged to MLflow.")


# Standalone execution
if __name__ == "__main__":
    evaluate_model()
