#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing script for Telco Customer Churn
Handles missing values, encoding, scaling, and train-test split.
Outputs four files: X_train, X_test, y_train, y_test
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocess():

    # -----------------------------
    # 1️⃣ Load raw data
    # -----------------------------
    raw_path = "data/raw/Churn Prediction DataSet.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    df = pd.read_csv(raw_path)

    # -----------------------------
    # 2️⃣ Handle missing values
    # -----------------------------
    # Convert TotalCharges to numeric (empty strings -> NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # -----------------------------
    # 3️⃣ Encode binary categorical columns
    # -----------------------------
    binary_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling',
        'Churn'
    ]

    mapping = {
        'Yes': 1,
        'No': 0,
        'Female': 0,
        'Male': 1
    }

    for col in binary_cols:
        df[col] = df[col].map(mapping)

    # -----------------------------
    # 4️⃣ Encode multi-category columns
    # -----------------------------
    multi_cat_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # -----------------------------
    # 5️⃣ Remove any remaining NaN values
    # -----------------------------
    df.fillna(0, inplace=True)

    # -----------------------------
    # 6️⃣ Scale numeric features
    # -----------------------------
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # -----------------------------
    # 7️⃣ Train-test split
    # -----------------------------
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # 8️⃣ MLOps Validation Check
    # -----------------------------
    if X_train.isna().sum().sum() > 0:
        raise ValueError("❌ NaN values detected in X_train")

    if X_test.isna().sum().sum() > 0:
        raise ValueError("❌ NaN values detected in X_test")

    if y_train.isna().sum() > 0:
        raise ValueError("❌ NaN values detected in y_train")

    if y_test.isna().sum() > 0:
        raise ValueError("❌ NaN values detected in y_test")

    # -----------------------------
    # 9️⃣ Save processed data
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # -----------------------------
    # 🔟 Logging
    # -----------------------------
    print("✅ Preprocessing complete")
    print("Processed files saved in: data/processed/")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    print("✅ Validation passed: No NaN values detected")


# -----------------------------
# Execute when run as script
# -----------------------------
if __name__ == "__main__":
    preprocess()

