# %%
df = pd.read_csv("../data/Loan_default.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

df.head()

# %%
df.info()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")

# %%
df.isnull().sum()

# %%
print("Number of duplicate rows:", df.duplicated().sum())

# %%
print(df["Default"].value_counts())

# %%
print(df["Default"].value_counts(normalize=True) * 100)

# %%
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Default")

plt.title("Loan Default Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Borrowers")

plt.show()

# %%
default_counts = df["Default"].value_counts()

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=default_counts.index,
    y=default_counts.values
)

plt.title("Loan Default Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Borrowers")

for i, value in enumerate(default_counts.values):
    ax.text(i, value + 3000, f"{value:,}", ha="center")

plt.show()

# %%
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Default",
    y="CreditScore"
)

plt.title("Credit Score vs Loan Default")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Credit Score")

plt.show()

# %%
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Default",
    y="Income"
)

plt.title("Income vs Loan Default")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Income")

plt.show()

# %%
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Default",
    y="LoanAmount"
)

plt.title("Loan Amount vs Loan Default")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Loan Amount")

plt.show()

# %%
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Default",
    y="DTIRatio"
)

plt.title("DTI Ratio vs Loan Default")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("DTI Ratio")

plt.show()

# %%
default_by_employment = (
    df.groupby("EmploymentType")["Default"]
      .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=default_by_employment.index,
    y=default_by_employment.values * 100
)

plt.title("Default Rate by Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30)

for i, value in enumerate(default_by_employment.values * 100):
    ax.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# %%
default_by_education = (
    df.groupby("Education")["Default"]
      .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=default_by_education.index,
    y=default_by_education.values * 100
)

plt.title("Default Rate by Education")
plt.xlabel("Education")
plt.ylabel("Default Rate (%)")

plt.ylim(0, (default_by_education.max() * 100) + 2)

for i, value in enumerate(default_by_education.values * 100):
    ax.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# %%
default_by_purpose = (
    df.groupby("LoanPurpose")["Default"]
      .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

ax = sns.barplot(
    x=default_by_purpose.index,
    y=default_by_purpose.values * 100
)

plt.title("Default Rate by Loan Purpose")
plt.xlabel("Loan Purpose")
plt.ylabel("Default Rate (%)")

plt.ylim(0, (default_by_purpose.max() * 100) + 2)

for i, value in enumerate(default_by_purpose.values * 100):
    ax.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# %%
default_by_mortgage = (
    df.groupby("HasMortgage")["Default"]
      .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=default_by_mortgage.index,
    y=default_by_mortgage.values * 100
)

plt.title("Default Rate by Mortgage Status")
plt.xlabel("Has Mortgage")
plt.ylabel("Default Rate (%)")

plt.ylim(0, (default_by_mortgage.max() * 100) + 2)

for i, value in enumerate(default_by_mortgage.values * 100):
    ax.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# %%
default_by_dependents = (
    df.groupby("HasDependents")["Default"]
      .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=default_by_dependents.index,
    y=default_by_dependents.values * 100
)

plt.title("Default Rate by Dependents")
plt.xlabel("Has Dependents")
plt.ylabel("Default Rate (%)")

plt.ylim(0, (default_by_dependents.max() * 100) + 2)

for i, value in enumerate(default_by_dependents.values * 100):
    ax.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# %%
default_by_cosigner = (
    df.groupby("HasCoSigner")["Default"]
      .mean()
      .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=default_by_cosigner.index,
    y=default_by_cosigner.values * 100
)

plt.title("Default Rate by Co-Signer Status")
plt.xlabel("Has Co-Signer")
plt.ylabel("Default Rate (%)")

plt.ylim(0, (default_by_cosigner.max() * 100) + 2)

for i, value in enumerate(default_by_cosigner.values * 100):
    ax.text(i, value + 0.2, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# %%
numeric_cols = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio"
]

correlation = df[numeric_cols + ["Default"]].corr()["Default"].sort_values(
    ascending=False
)

print(correlation)

# %%
X = df.drop(columns=["Default", "LoanID"])
y = df["Default"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# %%
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

print("Preprocessing pipeline created successfully!")

# %%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

print("Logistic Regression pipeline created!")

# %%
logistic_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")

# %%
y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

disp.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_prob):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

# %%
df.describe().T

# %%
categorical_columns = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner"
]

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].unique())

# %%
print("Age below 18:", (df["Age"] < 18).sum())
print("Negative income:", (df["Income"] < 0).sum())
print("Negative loan amount:", (df["LoanAmount"] < 0).sum())
print("Credit score below 0:", (df["CreditScore"] < 0).sum())
print("Interest rate below 0:", (df["InterestRate"] < 0).sum())
print("DTI below 0:", (df["DTIRatio"] < 0).sum())

# %%
logistic_results = {
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

logistic_results

# %%



