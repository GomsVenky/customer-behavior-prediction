import pandas as pd

from src.models.clv_predictor import (
    load_clv_model_and_config,
    predict_clv,
)


def test_load_clv_model_and_config():
    """
    Test that the saved CLV model and configuration load correctly.
    """
    model, config = load_clv_model_and_config()

    assert model is not None
    assert config["model"] == "RandomForestRegressor"

    assert config["features"] == [
        "Income",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "Customer_Tenure",
    ]

    assert config["target"] == "CLV_Proxy"


def test_predict_clv():
    """
    Test that the CLV predictor returns a numeric prediction.
    """
    customer = pd.DataFrame(
        [
            {
                "Income": 50000,
                "NumWebPurchases": 3,
                "NumCatalogPurchases": 2,
                "NumStorePurchases": 4,
                "Customer_Tenure": 12,
            }
        ]
    )

    result = predict_clv(customer)

    assert "Predicted_CLV_Proxy" in result.columns
    assert len(result) == 1
    assert result.iloc[0]["Predicted_CLV_Proxy"] >= 0


def test_clv_missing_features():
    """
    Test that incomplete CLV input raises a clear error.
    """
    customer = pd.DataFrame(
        [
            {
                "Income": 50000,
                "NumWebPurchases": 3,
            }
        ]
    )

    try:
        predict_clv(customer)
        assert False, "Expected ValueError for missing CLV features"

    except ValueError as error:
        assert "Missing CLV features" in str(error)