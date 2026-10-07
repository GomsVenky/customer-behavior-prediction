import pandas as pd

from src.data.cleaning import clean_marketing_data
from src.features.engineering import engineer_customer_features
from src.models.predictor import load_model_and_config


def test_target_not_used_as_model_feature():
    """
    Ensure the prediction target is never included in model inputs.
    """
    _, config = load_model_and_config()

    model_features = config["features"]

    assert "Response" not in model_features


def test_identifier_not_used_as_model_feature():
    """
    Ensure customer ID is excluded from model training features.
    """
    _, config = load_model_and_config()

    model_features = config["features"]

    assert "ID" not in model_features


def test_raw_date_columns_removed_after_feature_engineering():
    """
    Ensure raw date/year columns are replaced by engineered features
    instead of being passed directly to the prediction model.
    """
    raw_data = pd.read_csv(
        "data/raw/marketing_campaign.csv",
        sep="\t",
    )

    cleaned_data = clean_marketing_data(raw_data)

    engineered_data = engineer_customer_features(cleaned_data)

    assert "ID" not in engineered_data.columns
    assert "Dt_Customer" not in engineered_data.columns
    assert "Year_Birth" not in engineered_data.columns

    assert "Age" in engineered_data.columns
    assert "Customer_Tenure" in engineered_data.columns