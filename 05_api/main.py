from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts loan default risk using a trained XGBoost model",
    version="1.0"
)

# Load model and feature columns once, when the API starts
model = joblib.load("04_models/loan_default_model.joblib")

with open("04_models/feature_columns.json") as f:
    feature_columns = json.load(f)


class LoanApplication(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(data: LoanApplication):
    # Convert incoming data to a DataFrame matching our training format
    raw = pd.DataFrame([data.dict()])

    # Apply the SAME preprocessing and feature engineering as training
    import sys
    sys.path.append("03_src")
    from data_processing import preprocess
    from train import engineer_features

    processed = preprocess(raw)
    processed = engineer_features(processed)

    # Align columns with what the model was trained on
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    probability = model.predict_proba(processed)[0][1]
    prediction = int(probability >= 0.5)

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "default_prediction": prediction,
        "default_probability": round(float(probability), 4),
        "risk_level": risk
    }