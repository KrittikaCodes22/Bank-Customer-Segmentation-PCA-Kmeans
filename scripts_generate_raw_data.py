"""
Generates a synthetic-but-realistic UNCLEANED bank customer transactions dataset,
structurally modeled on the popular Kaggle "Bank Customer Segmentation" dataset
(TransactionID, CustomerID, CustomerDOB, CustGender, CustLocation,
CustAccountBalance, TransactionDate, TransactionTime, TransactionAmount).

Four latent customer archetypes are baked in (Premium / Regular / Budget / Dormant)
so that PCA + K-Means later recovers meaningful, presentable segments -- while the
raw file itself is left deliberately MESSY for the Day-2/3 data cleaning exercise:
- missing values
- inconsistent gender labels / casing
- inconsistent date formats
- balance/amount stored as strings with currency symbols & commas
- duplicate rows
- whitespace / casing issues in location
- unrealistic DOBs and outlier amounts
"""

import numpy as np
import pandas as pd
import random

rng = np.random.default_rng(42)
random.seed(42)

N_CUSTOMERS = 800
N_TRANSACTIONS = 6000

cities = ["MUMBAI", "Mumbai", "mumbai ", "DELHI", "Delhi", "BANGALORE", "Bangalore",
          "Bengaluru", "CHENNAI", "Chennai", "KOLKATA", "Kolkata", "HYDERABAD",
          "Hyderabad", "PUNE", "Pune", "AHMEDABAD", "Ahmedabad", "JAIPUR", "Jaipur",
          "LUCKNOW", np.nan, "  Surat", "NAGPUR"]

# ---- Latent customer archetypes (hidden ground truth used to drive realistic patterns) ----
SEGMENTS = {
    "Premium":  {"weight": 0.15, "balance_scale": 180000, "amount_scale": 4500, "freq_weight": 3.0, "recency_bias": 0},
    "Regular":  {"weight": 0.45, "balance_scale": 55000,  "amount_scale": 1600, "freq_weight": 1.6, "recency_bias": 0},
    "Budget":   {"weight": 0.25, "balance_scale": 12000,  "amount_scale": 500,  "freq_weight": 1.0, "recency_bias": 0},
    "Dormant":  {"weight": 0.15, "balance_scale": 20000,  "amount_scale": 700,  "freq_weight": 0.4, "recency_bias": 200},
}
seg_names = list(SEGMENTS.keys())
seg_weights = [SEGMENTS[s]["weight"] for s in seg_names]

customer_ids = [f"C{100000+i}" for i in range(N_CUSTOMERS)]
cust_segment = {}
cust_gender = {}
cust_dob = {}
cust_location = {}
cust_balance = {}

for cid in customer_ids:
    seg = random.choices(seg_names, weights=seg_weights, k=1)[0]
    cust_segment[cid] = seg
    params = SEGMENTS[seg]

    gender_clean = random.choice(["Male", "Female"])
    # inject messiness in the *raw* label (casing/abbreviation), 12% missing
    if random.random() < 0.12:
        cust_gender[cid] = np.nan
    else:
        pool = ["M", "Male", "male", "m"] if gender_clean == "Male" else ["F", "Female", "female", "f"]
        cust_gender[cid] = random.choice(pool)

    # DOB: mostly realistic, some corrupted/unrealistic entries
    if random.random() < 0.05:
        dob = random.choice(["1/1/1800", "01-01-1900", "2035-01-01", np.nan, "0000-00-00"])
    else:
        year = rng.integers(1955, 2005)
        month = rng.integers(1, 13)
        day = rng.integers(1, 28)
        fmt = random.choice(["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%y"])
        dob = pd.Timestamp(year=int(year), month=int(month), day=int(day)).strftime(fmt)
    cust_dob[cid] = dob

    cust_location[cid] = random.choice(cities)

    # balance driven by segment, then messiness applied on top
    bal = round(float(rng.gamma(shape=2.0, scale=params["balance_scale"] / 2.0)), 2)
    if random.random() < 0.15:
        bal_val = np.nan
    else:
        style = random.random()
        if style < 0.4:
            bal_val = bal
        elif style < 0.7:
            bal_val = f"{bal:,.2f}"
        else:
            bal_val = f"Rs. {bal:,.2f}"
    cust_balance[cid] = bal_val

# customer sampling weights for transactions -> drives "Frequency" per segment
cust_weight = np.array([SEGMENTS[cust_segment[cid]]["freq_weight"] for cid in customer_ids], dtype=float)
cust_weight = cust_weight / cust_weight.sum()

# ---- generate transactions ----
rows = []
for i in range(N_TRANSACTIONS):
    cid = np.random.choice(customer_ids, p=cust_weight)
    seg = cust_segment[cid]
    params = SEGMENTS[seg]
    txn_id = f"T{500000+i}"

    # Dormant customers get older transaction dates (higher recency), others spread across the year
    max_day_offset = max(365 - params["recency_bias"], 30)
    day_offset = int(rng.integers(0, max_day_offset))
    date_choice = pd.Timestamp("2024-01-01") + pd.to_timedelta(day_offset, unit="D")
    date_fmt = random.choice(["%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y"])
    txn_date = date_choice.strftime(date_fmt)

    txn_time = f"{rng.integers(0,24):02d}{rng.integers(0,60):02d}{rng.integers(0,60):02d}"

    amt = round(float(rng.gamma(shape=1.5, scale=params["amount_scale"])), 2)
    if random.random() < 0.02:
        amt = -abs(amt)  # erroneous negative amount
    if random.random() < 0.01:
        amt = amt * 100  # extreme outlier / data-entry error
    if random.random() < 0.1:
        amt_val = f"INR {amt:,.2f}"
    else:
        amt_val = amt
    if random.random() < 0.03:
        amt_val = np.nan

    rows.append({
        "TransactionID": txn_id,
        "CustomerID": cid,
        "CustomerDOB": cust_dob[cid],
        "CustGender": cust_gender[cid],
        "CustLocation": cust_location[cid],
        "CustAccountBalance": cust_balance[cid],
        "TransactionDate": txn_date,
        "TransactionTime": txn_time,
        "TransactionAmount (INR)": amt_val,
    })

df = pd.DataFrame(rows)

# introduce exact duplicate rows
dup_frac = 0.03
n_dupes = int(len(df) * dup_frac)
dupes = df.sample(n=n_dupes, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# shuffle
df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

df.to_csv("data/raw/bank_transactions_raw.csv", index=False)
print("Saved raw uncleaned dataset:", df.shape)
print(df.isna().sum())
print("\nLatent segment distribution (hidden ground truth, NOT in the CSV):")
print(pd.Series(cust_segment).value_counts())
