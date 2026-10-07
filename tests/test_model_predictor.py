import pandas as pd

from src.models.predictor import (
    load_model_and_config,
    predict_customer_response,
)


def test_load_model_and_config():
    model, config = load_model_and_config()

    assert model is not None
    assert "features" in config
    assert "threshold" in config
    assert len(config["features"]) == 38


def test_predict_customer_response():
    _, config = load_model_and_config()

    features = pd.DataFrame(
        [
            [0] * len(config["features"]),
        ],
        columns=config["features"],
    )

    result = predict_customer_response(features)

    assert "response_probability" in result.columns
    assert "predicted_response" in result.columns
    assert "prediction_label" in result.columns

    assert len(result) == 1
    assert 0 <= result.loc[0, "response_probability"] <= 1
    assert result.loc[0, "predicted_response"] in [0, 1]