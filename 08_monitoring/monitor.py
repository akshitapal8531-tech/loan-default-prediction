import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUMERIC_COLS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]


def psi(expected, actual, bins=10):
    """Population Stability Index: how much a feature's distribution has shifted."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value):
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate drift"
    return "SIGNIFICANT DRIFT"


def drift_report(reference, current):
    rows = []
    for col in NUMERIC_COLS:
        value = psi(reference[col], current[col])
        rows.append({"feature": col, "psi": round(value, 4), "status": status(value)})
    return pd.DataFrame(rows)


# Load the data and rebuild the same train/test split used for training
df = pd.read_csv(os.path.join(BASE, "01_data", "Loan_default.csv"))
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Default"]
)

# Check 1: normal incoming data (the test set) should look like training data
print("CHECK 1: test data vs training data (expect no drift)")
normal = drift_report(train, test)
print(normal.to_string(index=False))

# Check 2: simulated drift, to show the monitor catches real changes
drifted = test.copy()
drifted["Income"] = drifted["Income"] * 0.8
drifted["InterestRate"] = drifted["InterestRate"] + 3
drifted["CreditScore"] = drifted["CreditScore"] - 40

print("\nCHECK 2: simulated drifted data (lower income, higher rates, lower credit scores)")
simulated = drift_report(train, drifted)
print(simulated.to_string(index=False))

# Save both reports
out_dir = os.path.join(BASE, "06_reports")
os.makedirs(out_dir, exist_ok=True)
normal.to_csv(os.path.join(out_dir, "drift_normal.csv"), index=False)
simulated.to_csv(os.path.join(out_dir, "drift_simulated.csv"), index=False)
print("\nSaved reports to 06_reports/")