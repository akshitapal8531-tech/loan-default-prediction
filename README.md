# Loan Default Prediction

An end-to-end machine learning project that predicts the probability a borrower will default on a loan. It covers data analysis, model comparison, training, a FastAPI prediction service, SHAP explainability and data drift monitoring.

## Project Goal

Lenders lose money when loans turn into non-performing assets (NPAs). The goal of this project is to flag high-risk applicants early so a lender can make better approval decisions. The business target behind the project was a 30% reduction in NPAs. This is a goal, not a measured result: it cannot be verified on a public dataset.

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

From the `05_api` folder:

```bash
cd 05_api
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. FastAPI also generates interactive documentation at `http://127.0.0.1:8000/docs`, where you can try the endpoint directly in the browser.

### 4. Make a prediction

Send a `POST` request to `/predict` with the applicant's details as JSON. The exact input fields are shown in the `/docs` page.

Example response:

```json
{
  "default_probability": 0.31,
  "risk_level": "Medium"
}
```

## Explainability and Monitoring

**SHAP:** The SHAP summary plot in `06_reports` shows which features push predictions toward default across the whole dataset. It is a global view. The API does not return per-prediction explanations.

**Drift monitoring:** The script in `08_monitoring` compares the distribution of new data against the training data and saves a report in `06_reports`. It is run manually. It is not scheduled and is not connected to the API.

## Limitations

- AUC of about 0.75 is below the original 0.85 target.
- The NPA reduction goal is not validated, since there is no real lending outcome data.
- SHAP is global only, and drift monitoring is a manual script, not a production system.
- The API has not been deployed to a public URL.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Uvicorn

## Author

Akshita Pal
