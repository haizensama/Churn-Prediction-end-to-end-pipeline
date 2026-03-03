import pandas as pd
import os

def main():
    input_path = "data/external/telco_churn.csv"
    output_path = "data/raw/Churn Prediction DataSet.csv"

    # Read original dataset
    df = pd.read_csv(input_path)

    # Ensure raw directory exists
    os.makedirs("data/raw", exist_ok=True)

    # Save as raw standardized file
    df.to_csv(output_path, index=False)

    print("Ingestion completed successfully.")

if __name__ == "__main__":
    main()
