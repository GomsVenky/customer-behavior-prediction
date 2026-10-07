import pandas as pd

from src.data.pipeline import run_data_pipeline


def test_run_data_pipeline(tmp_path):
    output_path = tmp_path / "customer_features.csv"

    processed_data = run_data_pipeline(
        output_path=output_path
    )

    assert output_path.exists()

    saved_data = pd.read_csv(output_path)

    assert len(processed_data) == 2237
    assert processed_data.shape[1] == 39

    assert "Total_Spending" in processed_data.columns
    assert "Total_Children" in processed_data.columns
    assert "Age" in processed_data.columns
    assert "Customer_Tenure" in processed_data.columns

    assert "ID" not in processed_data.columns
    assert "Dt_Customer" not in processed_data.columns
    assert "Year_Birth" not in processed_data.columns

    assert processed_data.isna().sum().sum() == 0

    assert saved_data.shape == processed_data.shape