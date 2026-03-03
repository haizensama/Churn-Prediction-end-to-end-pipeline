import pandas as pd

def main():
    df = pd.read_csv("data/raw/Churn Prediction DataSet.csv")

    assert df.isnull().sum().sum() == 0, "Missing values detected"
    assert df.shape[0] > 0, "Dataset is empty"

    print("Validation passed")

if __name__ == "__main__":
    main()
