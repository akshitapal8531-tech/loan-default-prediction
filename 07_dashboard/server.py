from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
SITE_DIR = BASE_DIR / "site"
REPORTS_DIR = BASE_DIR.parent / "06_reports"
PREDICTION_API = "http://127.0.0.1:8001/predict"

app = FastAPI(title="Loan Default Risk Dashboard")


@app.post("/api/predict")
async def predict(request: Request):
    payload = await request.json()
    try:
        response = requests.post(PREDICTION_API, json=payload, timeout=10)
    except requests.RequestException:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Prediction API unavailable",
                "detail": "The prediction service is not responding. Please try again later.",
            },
        )
    try:
        answer = response.json()
    except ValueError:
        return JSONResponse(
            status_code=502,
            content={
                "error": "Invalid prediction response",
                "detail": "The prediction service returned an unreadable response.",
            },
        )
    return JSONResponse(content=answer, status_code=response.status_code)


app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")
