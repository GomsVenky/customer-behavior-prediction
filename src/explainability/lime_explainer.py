from functools import lru_cache

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.models.predictor import load_model_and_config


@lru_cache(maxsize=1)
def _load_lime_resources() -> tuple:
    """
    Load the prediction model, configuration, and processed training data
    required by the LIME explainer.
    """
    model, config = load_model_and_config()

    processed_data = pd.read_csv(
        "data/processed/customer_features.csv"
    )

    feature_names = config["features"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in processed_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Processed data is missing features: {missing_features}"
        )

    training_data = processed_data[feature_names].astype(float)

    explainer = LimeTabularExplainer(
        training_data=training_data.to_numpy(),
        feature_names=feature_names,
        class_names=["Unlikely to Respond", "Likely to Respond"],
        mode="classification",
        random_state=42,
    )

    return model, config, explainer


def explain_customer_with_lime(
    features: pd.DataFrame,
    num_features: int = 5,
) -> pd.DataFrame:
    """
    Explain a single customer's response prediction using LIME.

    Parameters
    ----------
    features:
        DataFrame containing one customer and the model's expected features.
    num_features:
        Number of influential features to return.

    Returns
    -------
    pd.DataFrame
        LIME explanation containing the feature rule and contribution.
    """
    model, config, explainer = _load_lime_resources()

    expected_features = config["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if len(features) != 1:
        raise ValueError(
            "LIME explanation requires exactly one customer."
        )

    customer = (
        features[expected_features]
        .astype(float)
        .iloc[0]
        .to_numpy()
    )

    explanation = explainer.explain_instance(
        data_row=customer,
        predict_fn=model.predict_proba,
        num_features=num_features,
    )

    lime_results = pd.DataFrame(
        explanation.as_list(),
        columns=["Feature_Rule", "LIME_Contribution"],
    )

    lime_results["Impact"] = np.where(
        lime_results["LIME_Contribution"] > 0,
        "Increases response likelihood",
        "Decreases response likelihood",
    )

    return lime_results