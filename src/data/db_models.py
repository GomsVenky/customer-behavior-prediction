from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.data.database import Base


class Prediction(Base):
    """
    Store customer response predictions.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    customer_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    response_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    predicted_response: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    prediction_label: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class CustomerSegment(Base):
    """
    Store customer segmentation results.
    """

    __tablename__ = "customer_segments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    customer_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    segment_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    segment_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )