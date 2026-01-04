import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import yaml
import os
import argparse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config["load_data"]["raw_dataset_csv"])
    
    train_df, test_df = train_test_split(
        df, 
        test_size=config["split_data"]["test_size"], 
        random_state=config["base"]["random_state"]
    )
    
    target = config["base"]["target_col"]
    train_x = train_df.drop([target], axis=1)
    test_x = test_df.drop([target], axis=1)
    train_y = train_df[target]
    test_y = test_df[target]
    
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    
    # mlflow.set_tracking_uri("http://localhost:5000") # Commented out for local default (mlruns folder)
    # If using DagsHub, the user would provide the URI and credentials.
    
    mlflow.set_experiment("My_New_Experiment")
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=config["base"]["random_state"])
        lr.fit(train_x, train_y)
        
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        
        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")
        
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        mlflow.sklearn.log_model(lr, "model")
        
        # Save model locally for FastAPI
        os.makedirs("models", exist_ok=True)
        import joblib
        joblib.dump(lr, "models/model.joblib")
        print("Model saved to models/model.joblib")

if __name__ == "__main__":
    train("params.yaml")
