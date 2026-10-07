from pathlib import Path

import pandas as pd


def load_marketing_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the customer marketing dataset from a TSV file.

    Parameters
    ----------
    file_path : str | Path
        Path to the marketing campaign dataset.

    Returns
    -------
    pd.DataFrame
        Loaded customer marketing data.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path, sep="\t")