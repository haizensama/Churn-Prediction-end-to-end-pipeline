#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import mlflow
from mlflow.tracking import MlflowClient

# -------------------------
# Authentication for DagsHub
# -------------------------
os.environ["MLFLOW_TRACKING_USERNAME"] = "haizensama"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "36c35c7a76fa8d3ab1be9fd7b1b0df25c5713e41"   # <-- replace with your token

# -------------------------
# MLflow tracking (DagsHub) - standard URI
# -------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/haizensama/mlops22ug2-0179.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Churn_Prediction"

def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
        return
    experiment_id = experiment.experiment_id

    # Find the run with the best_model tag (set during training)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string='tags.best_model != ""',
        max_results=1
    )
    if not runs:
        print("⚠️ No run with 'best_model' tag found. Make sure training completed and set the tag.")
        return

    best_run = runs[0]
    best_model_name = best_run.data.tags["best_model"]
    best_run_id = best_run.info.run_id
    print(f"✅ Best model identified: {best_model_name} (run_id: {best_run_id})")

    # ------------------------------------------------------------------
    # Ensure the model is registered in the Model Registry
    # ------------------------------------------------------------------
    try:
        # Check if model is already registered
        client.get_registered_model(best_model_name)
        print(f"ℹ️ Model '{best_model_name}' is already registered.")
    except mlflow.exceptions.MlflowException:
        # Model not registered – register it from the best run's artifact
        print(f"📦 Registering model '{best_model_name}' from run {best_run_id}...")
        model_uri = f"runs:/{best_run_id}/{best_model_name}"
        mlflow.register_model(model_uri=model_uri, name=best_model_name)

    # ------------------------------------------------------------------
    # Get the latest version and promote to Production
    # ------------------------------------------------------------------
    latest_versions = client.get_latest_versions(best_model_name)
    if not latest_versions:
        print(f"❌ No versions found for model '{best_model_name}'. Registration may have failed.")
        return

    # Latest version is the one with the highest version number
    latest_version = latest_versions[-1].version
    print(f"ℹ️ Latest version of '{best_model_name}': {latest_version}")

    # Transition to Production, archiving any existing Production versions
    client.transition_model_version_stage(
        name=best_model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"🚀 Promoted {best_model_name} version {latest_version} to Production")

if __name__ == "__main__":
    main()
