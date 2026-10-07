from unittest.mock import MagicMock

from src.data.db_models import Prediction
from src.data.db_service import save_prediction


def test_save_prediction():
    mock_db = MagicMock()

    result = save_prediction(
        db=mock_db,
        customer_id=101,
        response_probability=0.75,
        predicted_response=1,
        prediction_label="Likely to Respond",
    )

    assert isinstance(result, Prediction)

    assert result.customer_id == 101
    assert result.response_probability == 0.75
    assert result.predicted_response == 1
    assert result.prediction_label == "Likely to Respond"

    mock_db.add.assert_called_once_with(result)
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(result)