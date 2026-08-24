"""
Reusable feature-engineering and clustering functions for the
Bank Customer Segmentation project. Mirrors notebooks 02 and 03.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "AvgTransactionAmount",
    "AccountBalance", "Age", "Gender_Male",
]


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clean transaction-level data into customer-level RFM + demographic features."""
    df = df.copy()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    snapshot_date = df["TransactionDate"].max() + pd.Timedelta(days=1)

    customer_df = df.groupby("CustomerID").agg(
        Recency=("TransactionDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionID", "count"),
        Monetary=("TransactionAmount (INR)", "sum"),
        AvgTransactionAmount=("TransactionAmount (INR)", "mean"),
        AccountBalance=("CustAccountBalance", "mean"),
        Age=("Age", "mean"),
        Gender=("CustGender", lambda x: x.mode().iloc[0]),
    ).reset_index()

    customer_df["Gender_Male"] = (customer_df["Gender"] == "Male").astype(int)
    return customer_df


def find_best_k(X_pca, k_range=range(2, 11)):
    """Return (best_k, silhouette_scores_dict) using silhouette score."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        scores[k] = silhouette_score(X_pca, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def fit_pipeline(customer_df: pd.DataFrame, n_clusters: int = None, variance_target: float = 0.90):
    """Fit StandardScaler -> PCA -> KMeans and return fitted objects + labeled DataFrame."""
    X = customer_df[FEATURE_COLS].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA().fit(X_scaled)
    explained = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(explained >= variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    if n_clusters is None:
        n_clusters, _ = find_best_k(X_pca)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    result = customer_df.copy()
    result["Cluster"] = labels
    return scaler, pca, kmeans, result
