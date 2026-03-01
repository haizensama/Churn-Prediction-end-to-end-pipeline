import os
import mlflow
from mlflow.tracking import MlflowClient

# MLflow credentials are read from environment variables
# Make sure these are set in your terminal before running the script
# export MLFLOW_TRACKING_URI=...
# export MLFLOW_TRACKING_USERNAME=...
# export MLFLOW_TRACKING_PASSWORD=...

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("test_experiment")  # Creates experiment if it doesn't exist

# Start a test run
with mlflow.start_run():
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)

# List experiments to confirm connection (MLflow 2.x)
client = MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print(exp.experiment_id, exp.name)
