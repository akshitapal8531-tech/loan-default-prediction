import pandas as pd

numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
                 "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]

binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]

education_order = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}

onehot_cols = ["EmploymentType", "MaritalStatus", "LoanPurpose"]


def load_data(path):
    """Load the raw CSV."""
    return pd.read_csv(path)


def preprocess(df):
    """Clean and encode the loan dataset. Returns a fully numeric DataFrame."""
    if "LoanID" in df.columns:
        df = df.drop(columns=["LoanID"])

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df["Education"] = df["Education"].map(education_order)

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    return df


if __name__ == "__main__":
    # Quick manual test when running this file directly
    df = load_data("../data/Loan_default.csv")
    df = preprocess(df)
    print(df.shape)
    print(df.dtypes)