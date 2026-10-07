from functools import lru_cache

import pandas as pd
import shap

from src.models.predictor import load_model_and_config


@lru_cache(maxsize=1)
def load_shap_explainer():
    """
    Load the prediction model and create a SHAP
    TreeExplainer for the LightGBM classifier.
    """
    model, _ = load_model_and_config()

    return shap.TreeExplainer(model)


def explain_customer(
    customer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate feature-level SHAP explanations
    for a single customer.
    """
    _, config = load_model_and_config()

    expected_features = config["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in customer.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing explanation features: {missing_features}"
        )

    model_input = customer[expected_features].copy()

    explainer = load_shap_explainer()

    shap_values = explainer.shap_values(model_input)

    # Handle SHAP output differences between versions.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    values = shap_values[0]

    explanation = pd.DataFrame(
        {
            "Feature": expected_features,
            "Feature_Value": model_input.iloc[0].values,
            "SHAP_Value": values,
        }
    )

    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    return explanation