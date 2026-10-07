from io import StringIO
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.schemas.customer import CustomerData
from src.data.database import get_db
from src.data.db_service import save_prediction
from src.models.predictor import load_model_and_config, predict_customer_response
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


_, config = load_model_and_config()
THRESHOLD = config["threshold"]


@router.post("")
def predict(
    customer: CustomerData,
    db: Annotated[Session, Depends(get_db)],
):
    """
    Predict campaign response for a single customer.
    """
    try:
        customer_data = pd.DataFrame([customer.model_dump()])

        result = predict_customer_response(customer_data)

        probability = float(
            result["response_probability"].iloc[0]
        )
        prediction = int(
            result["predicted_response"].iloc[0]
        )
        label = result["prediction_label"].iloc[0]

        save_prediction(
            db=db,
            response_probability=probability,
            predicted_response=prediction,
            prediction_label=label,
        )

        logger.info(
            "Prediction completed: predicted_response=%s, "
            "response_probability=%.6f, label=%s",
             prediction,
             probability,
             label,
        )

        return {
            "response_probability": probability,
            "predicted_response": prediction,
            "prediction_label": label,
            "threshold_used": THRESHOLD,
        }

    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    
@router.post("/batch")
async def predict_batch(file: Annotated[UploadFile, File()]):
    """
    Generate campaign-response predictions for customers in an uploaded CSV.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()

        customer_data = pd.read_csv(StringIO(contents.decode("utf-8")))

        if customer_data.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty.",
            )

        result = predict_customer_response(customer_data)

        output = StringIO()
        result.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="customer_predictions.csv"'
            },
        )

    except HTTPException:
        raise

    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV file must use UTF-8 encoding.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc