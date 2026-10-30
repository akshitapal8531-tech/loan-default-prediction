import pandas as pd
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from data_processing import load_data, preprocess


def engineer_features(df):
    """Create additional derived features."""
    df["LoanToIncome"] = df["LoanAmount"] / (df["Income"] + 1)
    df["CreditUtilization"] = df["NumCreditLines"] / (df["MonthsEmployed"] + 1)
    df["IncomePerYear"] = df["Income"] / (df["Age"] + 1)
    df["InterestBurden"] = df["InterestRate"] * df["LoanAmount"] / (df["Income"] + 1)
    return df


def train_model():
    # 1. Load and preprocess
    df = load_data("../01_data/Loan_default.csv")
    df = preprocess(df)
    df = engineer_features(df)

    # 2. Split features and target
    X = df.drop(columns=["Default"])
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    print("XGBoost AUC:", auc)
    print(classification_report(y_test, y_pred))

    # 5. Save model and feature columns
    joblib.dump(model, "../04_models/loan_default_model.joblib")

    feature_columns = list(X_train.columns)
    with open("../04_models/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)

    print("Model and feature columns saved to 04_models/")


if __name__ == "__main__":
    train_model()