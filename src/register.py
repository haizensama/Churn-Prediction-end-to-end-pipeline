import os
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/haizensama/mlops22ug2-0179.mlflow"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Churn_Prediction"
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]

def main():
    client = MlflowClient()

    # Ensure experiment exists
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"Experiment '{EXPERIMENT_NAME}' created with id {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment '{EXPERIMENT_NAME}' with id {experiment_id}")

    # Register latest run for each model
    for model_name in MODELS:
        # Search runs by model_name tag
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags.model_name = "{model_name}"',
            order_by=["metrics.accuracy DESC"],
            max_results=1
        )

        if not runs:
            print(f"No runs found for model: {model_name}")
            continue

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/{model_name}"

        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(model_name)
            print(f"Registered model '{model_name}' created.")
        except mlflow.exceptions.MlflowException:
            # Already exists
            pass

        # Register the model version
        client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        print(f"Model '{model_name}' registered from run {run_id}")

if __name__ == "__main__":
    main()
