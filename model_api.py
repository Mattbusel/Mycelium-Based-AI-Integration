import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


MODEL_PATH = 'mycelium_growth_model.pkl'
model = joblib.load(MODEL_PATH)


app = FastAPI()


class PredictionRequest(BaseModel):
    time: float
    temperature: float
    humidity: float
    ph: float
    light_intensity: float
    co2_level: float
    substrate_type: float  


@app.post("/predict/")
def predict(data: PredictionRequest):
  
    features = np.array([
        [
            data.time,
            data.temperature,
            data.humidity,
            data.ph,
            data.light_intensity,
            data.co2_level,
            data.substrate_type
        ]
    ])
    
 
    prediction = model.predict(features)
    
   
    return {"predicted_growth": prediction[0]}


@app.get("/")
def read_root():
    return {"message": "Mycelium Growth Model API is running"}
