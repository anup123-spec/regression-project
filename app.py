from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
bundle = joblib.load("model.pkl")
model, scaler = bundle["model"], bundle["scaler"]

class Features(BaseModel):
    features: list[float]  # 8 values matching California housing features

@app.post("/predict")
def predict(data: Features):
    X = scaler.transform([data.features])
    pred = model.predict(X)[0]
    return {"prediction": round(pred, 4)}