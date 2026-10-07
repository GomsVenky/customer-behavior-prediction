import pandas as pd

from src.segmentation.customer_segmentation import (
    SEGMENT_FEATURES,
    create_customer_segments,
)


def test_create_customer_segments():
    data = pd.DataFrame(
        {
            "Income": [30000, 35000, 70000, 80000],
            "Total_Spending": [100, 150, 1200, 1500],
            "Total_Children": [2, 1, 0, 0],
            "Age": [35, 40, 50, 55],
            "Recency": [60, 50, 20, 10],
            "NumWebPurchases": [1, 2, 7, 8],
            "NumCatalogPurchases": [0, 1, 5, 6],
            "NumStorePurchases": [2, 3, 8, 10],
        }
    )

    result = create_customer_segments(data)

    assert len(result) == 4
    assert "Segment_ID" in result.columns
    assert "Segment_Name" in result.columns

    assert result["Segment_ID"].nunique() == 2

    assert set(result["Segment_Name"]).issubset(
        {"High-Value Customers", "Lower-Value Customers"}
    )


def test_create_customer_segments_missing_feature():
    data = pd.DataFrame(
        {
            feature: [1, 2, 3, 4]
            for feature in SEGMENT_FEATURES
            if feature != "Income"
        }
    )

    try:
        create_customer_segments(data)
        assert False, "Expected ValueError for missing feature"
    except ValueError as error:
        assert "Income" in str(error)