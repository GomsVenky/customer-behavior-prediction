from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

MODEL_PATH = MODEL_DIR / "final_lightgbm_model.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.pkl"


def load_model_and_config():
    """
    Load the trained LightGBM model and its configuration.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {CONFIG_PATH}")

    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    return model, config


def predict_customer_response(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict campaign response for customer records.

    Parameters
    ----------
    features : pd.DataFrame
        Feature-engineered customer data containing the
        exact features expected by the trained model.

    Returns
    -------
    pd.DataFrame
        Original features with prediction probability,
        predicted class, and prediction label.
    """
    model, config = load_model_and_config()

    expected_features = config["features"]
    threshold = config.get("threshold", 0.2)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    model_input = features[expected_features].copy()

    probabilities = model.predict_proba(model_input)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = features.copy()
    results["response_probability"] = probabilities
    results["predicted_response"] = predictions
    results["prediction_label"] = pd.Series(predictions, index=results.index).map(
        {
            0: "Unlikely to Respond",
            1: "Likely to Respond",
        }
    )

    return results