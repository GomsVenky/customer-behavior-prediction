from src.utils.config import load_config


def test_load_config():
    """
    Test that the project YAML configuration
    loads with the expected core settings.
    """
    config = load_config()

    assert config["project"]["name"] == "Customer Behavior Prediction"
    assert config["project"]["random_state"] == 42
    assert config["project"]["reference_year"] == 2026

    assert config["prediction"]["model_name"] == "LightGBM"
    assert config["prediction"]["threshold"] == 0.20

    assert config["segmentation"]["n_clusters"] == 2