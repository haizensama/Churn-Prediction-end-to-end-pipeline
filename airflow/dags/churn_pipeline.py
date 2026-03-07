#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime
import subprocess

# Add project root so DAG can import src scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from airflow import DAG
from airflow.operators.python import PythonOperator

# Import scripts that do NOT depend on MLflow
from src.ingestion import main as ingest_data
from src.validation import main as validate_data
from src.preprocessing import preprocess

# Path to the MLflow venv Python
ML_PYTHON = "/home/haizen/Desktop/churn-mlops-project/venv/bin/python"

# Helper function to run scripts in MLflow venv
def run_in_mlflow(script_path):
    project_root = "/home/haizen/Desktop/churn-mlops-project"
    subprocess.run(
        [ML_PYTHON, script_path],
        cwd=project_root,
        check=True
    )


# -----------------------------
# Default DAG args
# -----------------------------
default_args = {
    "owner": "haizen",
    "depends_on_past": False,
    "start_date": datetime(2026, 3, 4),
    "retries": 1,
}

# -----------------------------
# DAG definition
# -----------------------------
with DAG(
    "churn_end_to_end",
    default_args=default_args,
    schedule_interval=None,  # manual trigger
    catchup=False,
) as dag:

    # Task 1: Data ingestion
    task_ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data
    )

    # Task 2: Data validation
    task_validate = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data
    )

    # Task 3: Feature engineering
    task_feature = PythonOperator(
        task_id="feature_engineering",
        python_callable=preprocess
    )

    # Task 4: Model training
    task_train = PythonOperator(
        task_id="model_training",
        python_callable=lambda: run_in_mlflow("/home/haizen/Desktop/churn-mlops-project/src/train.py")
    )

    # Task 5: Model evaluation
    task_evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=lambda: run_in_mlflow("/home/haizen/Desktop/churn-mlops-project/src/evaluate.py")
    )

    # Task 6: Model registration
    task_register = PythonOperator(
        task_id="model_registration",
        python_callable=lambda: run_in_mlflow("/home/haizen/Desktop/churn-mlops-project/src/register.py")
    )

    # -----------------------------
    # Task dependencies
    # -----------------------------
    task_ingest >> task_validate >> task_feature >> task_train >> task_evaluate >> task_register
