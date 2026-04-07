# 🚀 Telco Customer Churn Prediction – End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-orange)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-red)
![FastAPI](https://img.shields.io/badge/FastAPI-API-teal)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)

---

## 📌 Project Overview

This project implements a **complete end-to-end MLOps pipeline** for predicting customer churn using machine learning.

It covers the **entire lifecycle**:

* Data ingestion & validation
* Feature engineering
* Model training with hyperparameter tuning
* Model evaluation & selection
* Model registration (MLflow)
* Workflow orchestration (Airflow)
* Data versioning (DVC)
* API deployment (FastAPI + Docker)

---

## 🏗️ Project Structure

```
churn-mlops-project/
│
├── airflow/                  # Airflow setup & DAGs
├── api/                      # FastAPI application
│   └── app_mlflow.py
│
├── data/
│   ├── external/             # Original dataset
│   ├── raw/                  # Ingested data (DVC)
│   └── processed/            # Preprocessed data (DVC)
│
├── evaluation/               # Evaluation outputs (plots)
├── models/                   # Trained models (DVC)
│
├── src/                      # Pipeline scripts
│   ├── ingestion.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── register.py
│
├── dvc.yaml                  # DVC pipeline
├── dvc.lock                  # Pipeline lock
├── Dockerfile                # Docker setup
├── requirements.txt          # API dependencies
├── requirements-airflow.txt  # Airflow dependencies
└── README.md
```

---

## ⚙️ Tech Stack

* **Python 3.12**
* **Scikit-learn**
* **XGBoost**
* **MLflow**
* **DVC**
* **Apache Airflow**
* **FastAPI**
* **Docker**

---

## 🔄 DVC Pipeline

### Pipeline Stages

1. Data Ingestion
2. Data Validation
3. Feature Engineering (Preprocessing)
4. Model Training (with hyperparameter tuning)
5. Model Evaluation
6. Model Registration

### ▶ Run Pipeline

```bash
dvc repro
```

### 📤 Push Data to Remote

```bash
dvc push
```

---

## 📊 MLflow Experiment Tracking

Tracks:

* Hyperparameters
* Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
* Models
* Evaluation artifacts

### ▶ Run MLflow UI (optional local)

```bash
mlflow ui
```

---

## 🔁 Airflow Orchestration

Automates the entire pipeline using DAGs.

### Tasks:

* ingestion
* validation
* feature_engineering
* training
* evaluation
* registration

### ▶ Start Airflow

```bash
airflow standalone
```

### 🌐 Access Airflow UI

http://localhost:8080

---

## 🌐 REST API (FastAPI)

### Endpoint

POST /predict

### 📥 Example Request

```json
{
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 840
}
```

### 📤 Example Response

```json
{
  "churn_probability": 0.82,
  "prediction": "Yes"
}
```

### ▶ Run API

```bash
uvicorn api.app_mlflow:app --reload
```

### 🌐 API Access

* API: http://localhost:8000
* Docs: http://localhost:8000/docs

---

## 🐳 Docker Deployment

### ▶ Build Image

```bash
docker build -t churn-api .
```

### ▶ Run Container

```bash
docker run -p 8000:8000 churn-api
```

---

## 📦 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd churn-mlops-project
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Setup

Place dataset here:

```
data/external/telco_churn.csv
```

Then run:

```bash
dvc repro
```

---

## 🧹 Cleanup Commands

### Remove DVC Stage

```bash
dvc remove <stage>
```

### Clean Git Files

```bash
git clean -fd
```

### Stop Docker Containers

```bash
docker stop $(docker ps -q)
```

### Remove Containers

```bash
docker rm $(docker ps -aq)
```

---

## 🔗 Access Points

| Service        | URL                        |
| -------------- | -------------------------- |
| Airflow UI     | http://localhost:8080      |
| FastAPI        | http://localhost:8000      |
| API Docs       | http://localhost:8000/docs |
| MLflow (local) | http://localhost:5000      |

---

## 🔐 Important Notes

❗ Do NOT commit:

* `venv/`
* `mlruns/`
* `data/` (use DVC instead)
* API tokens / credentials

---

## 📈 Key Features

✔ End-to-end automated ML pipeline
✔ Reproducible experiments with DVC
✔ Remote experiment tracking (DagsHub + MLflow)
✔ Workflow orchestration with Airflow
✔ Production-ready API with FastAPI
✔ Dockerized deployment

---

## 🚧 Future Improvements

* CI/CD pipeline integration
* Model monitoring & drift detection
* Automated retraining
* Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

**Charith Hewage**
MLOps Pipeline Project lead

**Malindi Ratnayake**
MLOps Pipeline Project Co-Lead
[Mali Ratnayake Github](https://github.com/maliratnayake)

**Achin Liyanage**
MLOps Pipeline Project Quality Tester
[Achin Liyanage Github]()

