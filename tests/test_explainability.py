import pandas as pd

from src.explainability.shap_explainer import explain_customer


def test_explain_customer():
    """
    Test that SHAP explanation is generated
    for a valid customer.
    """
    data = pd.read_csv(
        "data/processed/customer_features.csv"
    )

    customer = data.head(1)

    explanation = explain_customer(customer)

    assert len(explanation) == 38

    assert "Feature" in explanation.columns
    assert "Feature_Value" in explanation.columns
    assert "SHAP_Value" in explanation.columns
    assert "Absolute_SHAP" in explanation.columns

    assert explanation["Absolute_SHAP"].is_monotonic_decreasing


def test_explanation_missing_features():
    """
    Test that incomplete model input raises
    a clear validation error.
    """
    customer = pd.DataFrame(
        [
            {
                "Income": 50000,
                "Age": 45,
            }
        ]
    )

    try:
        explain_customer(customer)
        assert False, "Expected ValueError for missing features"

    except ValueError as error:
        assert "Missing explanation features" in str(error)