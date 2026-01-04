from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, model_validator
import uvicorn
import os

app = FastAPI()

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    @model_validator(mode='before')
    @classmethod
    def normalize_keys(cls, data: any) -> any:
        if isinstance(data, dict):
            return {k.replace(' ', '_'): v for k, v in data.items()}
        return data

# Load the model
model_path = "models/model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

@app.get("/")
def read_root():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict")
def predict(features: WineFeatures):
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    data = features.dict()
    df = pd.DataFrame([data])
    # Map underscores back to spaces to match training feature names
    df.columns = [col.replace('_', ' ') for col in df.columns]
    
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
