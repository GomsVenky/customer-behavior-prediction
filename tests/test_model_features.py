from pathlib import Path

from src.data.cleaning import clean_marketing_data
from src.data.loader import load_marketing_data
from src.features.engineering import engineer_customer_features

EXPECTED_FEATURES = [
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "Z_CostContact",
    "Z_Revenue",
    "Total_Spending",
    "Total_Children",
    "Age",
    "Customer_Tenure",
    "Education_Basic",
    "Education_Graduation",
    "Education_Master",
    "Education_PhD",
    "Marital_Status_Alone",
    "Marital_Status_Divorced",
    "Marital_Status_Married",
    "Marital_Status_Single",
    "Marital_Status_Together",
    "Marital_Status_Widow",
    "Marital_Status_YOLO",
]


def test_model_features():
    data_path = Path("data/raw/marketing_campaign.csv")

    raw_df = load_marketing_data(data_path)
    cleaned_df = clean_marketing_data(raw_df)
    features_df = engineer_customer_features(cleaned_df)

    actual_features = [
        column for column in features_df.columns
        if column != "Response"
    ]

    assert actual_features == EXPECTED_FEATURES