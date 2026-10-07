from sqlalchemy.orm import Session

from src.data.db_models import CustomerSegment, Prediction


def save_prediction(
    db: Session,
    response_probability: float,
    predicted_response: int,
    prediction_label: str,
    customer_id: int | None = None,
) -> Prediction:
    """
    Save a customer response prediction to the database.
    """
    prediction = Prediction(
        customer_id=customer_id,
        response_probability=response_probability,
        predicted_response=predicted_response,
        prediction_label=prediction_label,
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction


def save_customer_segment(
    db: Session,
    customer_id: int,
    segment_id: int,
    segment_name: str,
) -> CustomerSegment:
    """
    Save a customer segment assignment to the database.
    """
    segment = CustomerSegment(
        customer_id=customer_id,
        segment_id=segment_id,
        segment_name=segment_name,
    )

    db.add(segment)
    db.commit()
    db.refresh(segment)

    return segment