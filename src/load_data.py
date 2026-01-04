import pandas as pd
import yaml
import os

def load_data(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    source = config["data_source"]["s3_source"]
    df = pd.read_csv(source, sep=",")
    
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    df.to_csv(raw_data_path, index=False)
    print(f"Data saved to {raw_data_path}")

if __name__ == "__main__":
    load_data("params.yaml")
