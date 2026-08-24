"""
Streamlit demo app - Bank Customer Segmentation using PCA + K-Means

Run with:
    streamlit run app/app.py

Loads the trained scaler, PCA transformer and K-Means model (produced by
notebooks/03_pca_kmeans_clustering.ipynb) and lets a user either:
  1. Explore the existing customer segments (from data/processed/customer_segments.csv), or
  2. Enter a new customer's RFM-style details and get their predicted segment.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "..", "data", "processed", "customer_segments.csv")

st.set_page_config(page_title="Bank Customer Segmentation", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(APP_DIR, "scaler.joblib"))
    pca = joblib.load(os.path.join(APP_DIR, "pca.joblib"))
    kmeans = joblib.load(os.path.join(APP_DIR, "kmeans_model.joblib"))
    feature_cols = joblib.load(os.path.join(APP_DIR, "feature_cols.joblib"))
    return scaler, pca, kmeans, feature_cols


@st.cache_data
def load_segment_data():
    return pd.read_csv(DATA_PATH)


st.title("🏦 Bank Customer Segmentation")
st.caption("Unsupervised segmentation using PCA (dimensionality reduction) + K-Means clustering")

scaler, pca, kmeans, feature_cols = load_artifacts()
df = load_segment_data()

tab1, tab2 = st.tabs(["📊 Explore Existing Segments", "🔮 Predict a New Customer's Segment"])

with tab1:
    st.subheader("Cluster Overview")
    n_clusters = df["Cluster"].nunique()
    st.write(f"Number of segments discovered: **{n_clusters}**")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Customers per segment")
        st.bar_chart(df["Cluster"].value_counts().sort_index())

    with col2:
        st.write("Average feature values per segment")
        profile = df.groupby("Cluster")[feature_cols].mean().round(2)
        st.dataframe(profile, use_container_width=True)

    st.write("Full customer segment table")
    st.dataframe(df, use_container_width=True, height=350)

with tab2:
    st.subheader("Enter Customer Details")
    st.write("Provide the customer's transaction behaviour to predict which segment they belong to.")

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency (days since last transaction)", min_value=0, value=30)
        frequency = st.number_input("Frequency (number of transactions)", min_value=1, value=5)
    with c2:
        monetary = st.number_input("Monetary (total transaction amount, INR)", min_value=0.0, value=5000.0)
        avg_amount = st.number_input("Average transaction amount (INR)", min_value=0.0, value=1000.0)
    with c3:
        balance = st.number_input("Account Balance (INR)", min_value=0.0, value=40000.0)
        age = st.number_input("Age", min_value=15, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Predict Segment", type="primary"):
        gender_male = 1 if gender == "Male" else 0
        input_row = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "AvgTransactionAmount": avg_amount,
            "AccountBalance": balance,
            "Age": age,
            "Gender_Male": gender_male,
        }])[feature_cols]

        X_scaled = scaler.transform(input_row)
        X_pca = pca.transform(X_scaled)
        cluster = int(kmeans.predict(X_pca)[0])

        st.success(f"Predicted Segment: **Cluster {cluster}**")

        st.write("How this customer compares to the segment's average profile:")
        comparison = pd.concat([
            input_row.T.rename(columns={0: "This Customer"}),
            df[df["Cluster"] == cluster][feature_cols].mean().round(2).rename("Segment Average")
        ], axis=1)
        st.dataframe(comparison, use_container_width=True)

st.divider()
st.caption(
    "Model artifacts (scaler, PCA, K-Means) are trained in notebooks/03_pca_kmeans_clustering.ipynb. "
    "Re-run the notebook after changing data to refresh these artifacts."
)
