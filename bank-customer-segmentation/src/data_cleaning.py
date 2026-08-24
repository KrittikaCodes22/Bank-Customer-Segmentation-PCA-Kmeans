"""
Reusable data-cleaning functions for the Bank Customer Segmentation project.
These mirror the step-by-step logic in notebooks/01_data_cleaning.ipynb so the
same cleaning pipeline can be imported and reused (e.g., from app.py or tests).
"""

import numpy as np
import pandas as pd


def clean_currency(x):
    """Convert messy currency strings ('Rs. 12,345.67', 'INR 500', 1200.0, NaN) to float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).replace("Rs.", "").replace("INR", "").replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return np.nan


def clean_gender(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    )
    return mapped.fillna("Unknown")


def clean_location(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.title()
    cleaned = cleaned.replace({"Nan": np.nan, "None": np.nan, "": np.nan})
    return cleaned.fillna("Unknown")


def parse_date(x):
    return pd.to_datetime(x, errors="coerce", dayfirst=True)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: raw messy transactions -> clean transaction-level DataFrame."""
    df = raw.drop_duplicates().copy()

    df["CustGender"] = clean_gender(df["CustGender"])
    df["CustLocation"] = clean_location(df["CustLocation"])

    df["CustAccountBalance"] = df["CustAccountBalance"].apply(clean_currency)
    df["TransactionAmount (INR)"] = df["TransactionAmount (INR)"].apply(clean_currency)
    df["TransactionAmount (INR)"] = df["TransactionAmount (INR)"].abs()

    upper_cap = df["TransactionAmount (INR)"].quantile(0.995)
    df["TransactionAmount (INR)"] = np.where(
        df["TransactionAmount (INR)"] > upper_cap, upper_cap, df["TransactionAmount (INR)"]
    )

    df["TransactionDate"] = df["TransactionDate"].apply(parse_date)
    df["CustomerDOB"] = df["CustomerDOB"].apply(parse_date)

    mask_bad_dob = (df["CustomerDOB"].dt.year < 1930) | (df["CustomerDOB"].dt.year > 2010)
    df.loc[mask_bad_dob, "CustomerDOB"] = pd.NaT

    reference_date = pd.Timestamp("2024-06-30")
    df["Age"] = ((reference_date - df["CustomerDOB"]).dt.days / 365.25)
    df["Age"] = df["Age"].fillna(df["Age"].median()).round(1)
    df = df[(df["Age"] >= 15) & (df["Age"] <= 90)]

    df["CustAccountBalance"] = df.groupby("CustGender")["CustAccountBalance"].transform(
        lambda s: s.fillna(s.median())
    )
    df["TransactionAmount (INR)"] = df["TransactionAmount (INR)"].fillna(
        df["TransactionAmount (INR)"].median()
    )

    df = df.dropna(subset=["TransactionDate"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/bank_transactions_raw.csv")
    clean = clean_transactions(raw)
    clean.to_csv("data/processed/bank_transactions_clean.csv", index=False)
    print("Cleaned dataset saved:", clean.shape)
