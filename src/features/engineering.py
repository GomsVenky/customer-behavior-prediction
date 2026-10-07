import pandas as pd


def engineer_customer_features(
    df: pd.DataFrame,
    reference_year: int = 2026,
) -> pd.DataFrame:
    """
    Create customer-level features for the behavior prediction pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer marketing data.
    reference_year : int, default=2026
        Year used to calculate age and customer tenure.

    Returns
    -------
    pd.DataFrame
        Feature-engineered customer data.
    """
    features = df.copy()

    # Total amount spent across all product categories.
    features["Total_Spending"] = (
        features["MntWines"]
        + features["MntFruits"]
        + features["MntMeatProducts"]
        + features["MntFishProducts"]
        + features["MntSweetProducts"]
        + features["MntGoldProds"]
    )

    # Total number of children in the household.
    features["Total_Children"] = (
        features["Kidhome"] + features["Teenhome"]
    )

    # Customer age at the chosen reference year.
    features["Age"] = reference_year - features["Year_Birth"]

    # Convert customer enrollment date to datetime.
    features["Dt_Customer"] = pd.to_datetime(
        features["Dt_Customer"],
        format="%d-%m-%Y",
    )

    # Customer tenure in years.
    features["Customer_Tenure"] = (
        reference_year - features["Dt_Customer"].dt.year
    )

    # Remove columns no longer required by the model.
    features = features.drop(
        columns=["ID", "Dt_Customer", "Year_Birth"]
    )

    # One-hot encode categorical features.
    features = pd.get_dummies(
        features,
        columns=["Education", "Marital_Status"],
        drop_first=True,
        dtype=int,
    )

    return features