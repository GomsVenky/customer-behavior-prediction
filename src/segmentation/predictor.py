from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "models_artifacts"

MODEL_PATH = ARTIFACT_DIR / "kmeans_model.pkl"
SCALER_PATH = ARTIFACT_DIR / "segmentation_scaler.pkl"
CONFIG_PATH = ARTIFACT_DIR / "segmentation_config.pkl"


def load_segmentation_artifacts():
    """
    Load the trained K-Means model, scaler,
    and segmentation configuration.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, scaler, config


def predict_customer_segment(
    customer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign customer records to trained customer segments.
    """
    model, scaler, config = load_segmentation_artifacts()

    expected_features = config["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in customer.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing segmentation features: {missing_features}"
        )

    model_input = customer[expected_features].copy()

    scaled_input = scaler.transform(model_input)

    segment_ids = model.predict(scaled_input)

    result = customer.copy()

    result["Segment_ID"] = segment_ids

    result["Segment_Name"] = [
        config["segment_names"][int(segment_id)]
        for segment_id in segment_ids
    ]

    return result