import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENT_FEATURES = [
    "Income",
    "Total_Spending",
    "Total_Children",
    "Age",
    "Recency",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]


def create_customer_segments(
    data: pd.DataFrame,
    n_clusters: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Segment customers using K-Means clustering.

    Returns the original data with cluster ID and
    business-readable segment name added.
    """
    missing_features = [
        feature
        for feature in SEGMENT_FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing segmentation features: {missing_features}"
        )

    result = data.copy()

    segment_data = result[SEGMENT_FEATURES].copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(segment_data)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    result["Segment_ID"] = kmeans.fit_predict(scaled_data)

    segment_spending = result.groupby("Segment_ID")[
        "Total_Spending"
    ].mean()

    high_value_cluster = segment_spending.idxmax()

    result["Segment_Name"] = result["Segment_ID"].apply(
        lambda cluster: (
            "High-Value Customers"
            if cluster == high_value_cluster
            else "Lower-Value Customers"
        )
    )

    return result