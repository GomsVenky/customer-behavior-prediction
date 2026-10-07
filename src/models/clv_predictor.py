from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "final_clv_rf_model.pkl"
CONFIG_PATH = MODEL_DIR / "clv_model_config.pkl"


def load_clv_model_and_config():
    """
    Load the trained CLV Random Forest model
    and its configuration.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CLV model not found: {MODEL_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"CLV config not found: {CONFIG_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, config


def predict_clv(features: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the historical customer value proxy.

    Note:
        CLV_Proxy represents historical Total_Spending
        in this project. It is not true future CLV.
    """
    model, config = load_clv_model_and_config()

    expected_features = config["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing CLV features: {missing_features}"
        )

    model_input = features[expected_features].copy()

    predictions = model.predict(model_input)

    result = features.copy()
    result["Predicted_CLV_Proxy"] = predictions

    return result