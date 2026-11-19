# Loan Default Prediction

An end-to-end machine learning project that predicts the probability a borrower will default on a loan. It covers data analysis, model comparison, training, a FastAPI prediction service, SHAP explainability and data drift monitoring.

## Project Goal

Lenders lose money when loans turn into non-performing assets (NPAs). The goal of this project is to flag high-risk applicants early so a lender can make better approval decisions. The business target behind the project was a 30% reduction in NPAs. This is a goal, not a measured result: it cannot be verified on a public dataset.

## Dashboard

The project includes a responsive offline-friendly web dashboard in `07_dashboard/site`.

- Risk Checker: submit an applicant profile and receive a default probability, risk level and plain-English advice
- Model Insights: view the saved SHAP summary plot and learn how to interpret it
- Data Drift: compare normal and simulated PSI reports in tables and bar charts
- Safe and Risky presets: quickly test both sides of the prediction model
- Modern glassmorphism UI: built with plain HTML, CSS and JavaScript

Run the FastAPI dashboard with:

```bash
python -m uvicorn server:app --app-dir 07_dashboard --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Results

| Item | Value |
|------|-------|
| Final model | XGBoost |
| AUC | ~0.75 |
| Original AUC target | 0.85 (not reached) |

XGBoost was chosen after comparing several models. The AUC is below the original target, which is common on a public, imbalanced credit dataset with limited features. Further gains would likely need better feature engineering, more data or hyperparameter tuning.

## What's Included

- **Data analysis and preprocessing:** exploring the data, handling missing values and encoding features
- **Model comparison:** several models compared, with XGBoost selected
- **Training script and saved model:** reproducible training, with the trained model stored in `04_models`
- **Prediction API:** a FastAPI service with a `/predict` endpoint that returns a default probability and a risk level
- **Interactive dashboard:** a plain HTML, CSS and JavaScript interface that displays predictions and reports
- **SHAP explainability:** a global feature-importance plot showing which features drive the model overall, saved in `06_reports`
- **Drift monitoring:** a script that compares new data against the training data and writes a report to `06_reports`

## Project Structure

```
loan-default-prediction/
├── 01_data/          # Dataset
├── 03_src/           # Preprocessing and training code
├── 04_models/        # Saved trained model
├── 05_api/           # FastAPI prediction service
├── 06_reports/       # SHAP plot and drift monitoring report
├── 07_dashboard/     # Web dashboard and FastAPI dashboard server
│   ├── server.py
│   └── site/         # index.html, style.css and script.js
├── 08_monitoring/    # Drift monitoring script
├── notebooks/        # Exploratory analysis and model comparison
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/akshitapal8531-tech/loan-default-prediction.git
cd loan-default-prediction
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the API

From the project root:

```bash
python -m uvicorn main:app --app-dir 05_api --host 127.0.0.1 --port 8001
```

The API will be available at `http://127.0.0.1:8001`. FastAPI also provides interactive documentation at `http://127.0.0.1:8001/docs`.

### 4. Run the dashboard

Keep the prediction API running in the first terminal, then run this command in a second terminal from the project root:

```bash
python -m uvicorn server:app --app-dir 07_dashboard --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

### 5. Make a prediction

Send a `POST` request to `http://127.0.0.1:8001/predict` with the applicant's details as JSON.

The dashboard forwards requests to this endpoint automatically. Direct example:

```json
{
  "Age": 35,
  "Income": 85000,
  "LoanAmount": 18000,
  "CreditScore": 720,
  "MonthsEmployed": 60,
  "NumCreditLines": 4,
  "InterestRate": 8.5,
  "LoanTerm": 36,
  "DTIRatio": 0.28,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "No",
  "HasDependents": "No",
  "HasCoSigner": "No",
  "LoanPurpose": "Other"
}
```

Example response:

```json
{
  "default_prediction": 0,
  "default_probability": 0.2867,
  "risk_level": "Low"
}
```

## Explainability and Monitoring

**SHAP:** The SHAP summary plot in `06_reports` shows which features push predictions toward default across the whole dataset. It is a global view. The API does not return per-prediction explanations.

**Drift monitoring:** The script in `08_monitoring` compares the distribution of new data against the training data and saves a report in `06_reports`. It is run manually. It is not scheduled and is not connected to the API.

## Limitations

- AUC of about 0.75 is below the original 0.85 target.
- The NPA reduction goal is not validated, since there is no real lending outcome data.
- SHAP is global only, and drift monitoring is a manual script, not a production system.
- The services are intended to run locally and have not been deployed to a public URL.
- The dashboard prediction API must be reachable on port `8001`.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Uvicorn, HTML, CSS, JavaScript

## Author

Akshita Pal
