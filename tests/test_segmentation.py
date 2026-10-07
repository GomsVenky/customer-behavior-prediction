import pandas as pd

from src.segmentation.predictor import predict_customer_segment


def test_predict_customer_segment():
    """
    Test that the saved segmentation model
    returns a valid customer segment.
    """
    customer = pd.DataFrame(
        [
            {
                "Income": 50000,
                "Total_Spending": 500,
                "Total_Children": 1,
                "Age": 45,
                "Recency": 30,
                "NumWebPurchases": 3,
                "NumCatalogPurchases": 2,
                "NumStorePurchases": 4,
            }
        ]
    )

    result = predict_customer_segment(customer)

    assert "Segment_ID" in result.columns
    assert "Segment_Name" in result.columns

    assert result.iloc[0]["Segment_ID"] in [0, 1]

    assert result.iloc[0]["Segment_Name"] in [
        "Lower-Value Customers",
        "High-Value Customers",
    ]


def test_segmentation_missing_features():
    """
    Test that missing required features
    raise a clear validation error.
    """
    customer = pd.DataFrame(
        [
            {
                "Income": 50000,
                "Age": 45,
            }
        ]
    )

    try:
        predict_customer_segment(customer)
        assert False, "Expected ValueError for missing features"

    except ValueError as error:
        assert "Missing segmentation features" in str(error)