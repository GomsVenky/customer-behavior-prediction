from pathlib import Path

from src.data.loader import load_marketing_data


def test_load_marketing_data():
    data_path = Path("data/raw/marketing_campaign.csv")

    df = load_marketing_data(data_path)

    assert not df.empty
    assert len(df) > 0
    assert "Response" in df.columns