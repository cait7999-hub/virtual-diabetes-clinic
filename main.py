from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import pandas as pd
import joblib

app = FastAPI(title="Virtual Diabetes Clinic API")

# Load trained model
model = joblib.load("models/model.joblib")

# Define patient input
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
    return {"status": "ok"}

@app.post("/predict")
def predict(patient: Patient):
    data = pd.DataFrame([patient.model_dump()])
    prediction = model.predict(data)[0]

    return {
        "predicted_progression": float(prediction)
    }

