import pandas as pd


def clean_marketing_data(
    df: pd.DataFrame,
    reference_year: int = 2026,
) -> pd.DataFrame:
    """
    Clean the customer marketing dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer marketing data.
    reference_year : int, default=2026
        Reference year used to identify unrealistic customer ages.

    Returns
    -------
    pd.DataFrame
        Cleaned customer marketing data.
    """
    cleaned_df = df.copy()

    # Remove duplicate customer records.
    cleaned_df = cleaned_df.drop_duplicates()

    # Fill missing Income values using the dataset median.
    cleaned_df["Income"] = cleaned_df["Income"].fillna(
        cleaned_df["Income"].median()
    )

    # Remove unrealistic age outliers.
    customer_age = reference_year - cleaned_df["Year_Birth"]

    cleaned_df = cleaned_df[
        customer_age <= 100
    ].copy()

    return cleaned_df