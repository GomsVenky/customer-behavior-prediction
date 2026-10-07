from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_marketing_data
from src.data.loader import load_marketing_data
from src.features.engineering import engineer_customer_features
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "marketing_campaign.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
)

CONFIG = load_config()
REFERENCE_YEAR = CONFIG["project"]["reference_year"]


def run_data_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    reference_year: int = REFERENCE_YEAR,
) -> pd.DataFrame:
    """
    Run the customer data preprocessing pipeline.

    The pipeline:
    1. Loads the raw marketing dataset.
    2. Cleans duplicate and missing records.
    3. Creates model-ready customer features.
    4. Saves the processed feature table.

    Parameters
    ----------
    input_path : str | Path
        Path to the raw marketing dataset.
    output_path : str | Path
        Path where the processed dataset will be saved.
    reference_year : int, default=2026
        Reference year used for age and customer-tenure calculations.

    Returns
    -------
    pd.DataFrame
        Processed customer feature table.
    """
    raw_data = load_marketing_data(input_path)

    cleaned_data = clean_marketing_data(raw_data)

    processed_data = engineer_customer_features(
        cleaned_data,
        reference_year=reference_year,
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_data.to_csv(
        output_path,
        index=False,
    )

    return processed_data


if __name__ == "__main__":
    data = run_data_pipeline()

    print("Data pipeline completed successfully.")
    print(f"Processed rows: {len(data)}")
    print(f"Processed columns: {len(data.columns)}")
    print(f"Saved to: {DEFAULT_OUTPUT_PATH}")