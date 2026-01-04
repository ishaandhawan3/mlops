import pandas as pd
import joblib
import yaml
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

def evaluate(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    test_data_path = config["split_data"]["test_path"]
    # For simplicity, if split_data hasn't run yet, we'll just use the raw data and split it here too
    # but strictly speaking, we should have a split script.
    # Let's add a split script too.
    
    # Check if test data exists, if not, we can't evaluate
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}. Please run split_data.py first.")
        return

    df = pd.read_csv(test_data_path)
    model = joblib.load("models/model.joblib")
    
    target = config["base"]["target_col"]
    test_x = df.drop([target], axis=1)
    test_y = df[target]
    
    predictions = model.predict(test_x)
    
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to reports/metrics.json: {metrics}")

if __name__ == "__main__":
    evaluate("params.yaml")
