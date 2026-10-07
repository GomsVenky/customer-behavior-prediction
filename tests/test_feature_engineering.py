from pathlib import Path

from src.data.cleaning import clean_marketing_data
from src.data.loader import load_marketing_data
from src.features.engineering import engineer_customer_features


def test_engineer_customer_features():
    data_path = Path("data/raw/marketing_campaign.csv")

    raw_df = load_marketing_data(data_path)
    cleaned_df = clean_marketing_data(raw_df)
    features_df = engineer_customer_features(cleaned_df)

    assert not features_df.empty

    assert "Total_Spending" in features_df.columns
    assert "Total_Children" in features_df.columns
    assert "Age" in features_df.columns
    assert "Customer_Tenure" in features_df.columns

    assert "Year_Birth" not in features_df.columns
    assert "Dt_Customer" not in features_df.columns

    assert features_df["Total_Spending"].notna().all()
    assert features_df["Total_Children"].notna().all()
    assert features_df["Age"].notna().all()
    assert features_df["Customer_Tenure"].notna().all()