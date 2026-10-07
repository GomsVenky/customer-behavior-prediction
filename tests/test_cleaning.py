import pandas as pd

from src.data.cleaning import clean_marketing_data


def test_clean_marketing_data_removes_age_outliers():
    test_data = pd.DataFrame(
        {
            "Year_Birth": [1990, 1980, 1900, 2000],
            "Income": [50000, 60000, None, 40000],
        }
    )

    cleaned_data = clean_marketing_data(
        test_data,
        reference_year=2026,
    )

    # 1900 gives age 126, so that row should be removed.
    assert len(cleaned_data) == 3

    ages = 2026 - cleaned_data["Year_Birth"]

    assert (ages <= 100).all()


def test_clean_marketing_data_imputes_missing_income():
    test_data = pd.DataFrame(
        {
            "Year_Birth": [1990, 1985, 1995],
            "Income": [40000, None, 60000],
        }
    )

    cleaned_data = clean_marketing_data(test_data)

    # Median of 40000 and 60000 is 50000.
    assert cleaned_data["Income"].isna().sum() == 0
    assert cleaned_data.loc[1, "Income"] == 50000