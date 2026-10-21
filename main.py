import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Virtual Diabetes Clinic API")

MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

# Load the trained model
model = joblib.load("models/model.joblib")


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }


@app.post("/predict")
def predict(patient: Patient):
    data = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(data)[0]

    return {
        "prediction": float(prediction)
    }
    


