import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def split_data(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df = pd.read_csv(raw_data_path)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=config["split_data"]["test_size"], 
        random_state=config["base"]["random_state"]
    )
    
    train_path = config["split_data"]["train_path"]
    test_path = config["split_data"]["test_path"]
    
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data split and saved to {train_path} and {test_path}")

if __name__ == "__main__":
    split_data("params.yaml")
