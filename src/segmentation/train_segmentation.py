from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.segmentation.customer_segmentation import SEGMENT_FEATURES
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = load_config()
N_CLUSTERS = CONFIG["segmentation"]["n_clusters"]
RANDOM_STATE = CONFIG["project"]["random_state"]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_features.csv"
)

ARTIFACT_DIR = PROJECT_ROOT / "models_artifacts"

SCALER_PATH = ARTIFACT_DIR / "segmentation_scaler.pkl"
MODEL_PATH = ARTIFACT_DIR / "kmeans_model.pkl"


def train_segmentation_model() -> None:
    """
    Train and save the customer segmentation scaler
    and K-Means model.
    """
    data = pd.read_csv(DATA_PATH)

    segment_data = data[SEGMENT_FEATURES].copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(segment_data)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
)

    clusters = kmeans.fit_predict(scaled_data)

    data["Segment_ID"] = clusters

    spending_by_cluster = data.groupby("Segment_ID")[
        "Total_Spending"
    ].mean()

    high_value_cluster = int(spending_by_cluster.idxmax())

    config = {
        "features": SEGMENT_FEATURES,
        "high_value_cluster": high_value_cluster,
        "segment_names": {
            high_value_cluster: "High-Value Customers",
            1 - high_value_cluster: "Lower-Value Customers",
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(
        config,
        ARTIFACT_DIR / "segmentation_config.pkl",
    )

    print("Segmentation artifacts saved successfully.")
    print(f"K-Means model: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    print(
        "Config:",
        ARTIFACT_DIR / "segmentation_config.pkl",
    )


if __name__ == "__main__":
    train_segmentation_model()