from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import os

MODEL_PATH = "model.pkl"
LOG_PATH = "predictions_log.csv"

app = FastAPI()

# Load model at startup
model = joblib.load(MODEL_PATH)

# Request body format
class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

def log_prediction(data: dict, prediction: int, probability: float):
    row = data.copy()
    row["prediction"] = prediction
    row["probability"] = probability
    row["timestamp"] = datetime.utcnow().isoformat()

    df = pd.DataFrame([row])

    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)

@app.get("/")
def home():
    return {"message": "Cloud Run CI/CD Active"}

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)

    log_prediction(data.dict(), pred, float(prob))

    return {"prediction": pred, "probability": float(prob)}
