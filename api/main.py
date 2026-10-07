from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.metadata import router as metadata_router
from api.routers.predictions import router as prediction_router

app = FastAPI(
    title="Customer Behavior Prediction API",
    description=(
        "API for predicting customer campaign response "
        "using LightGBM."
    ),
    version="1.0.0",
)

app.include_router(prediction_router)
app.include_router(health_router)
app.include_router(metadata_router)


@app.get("/")
def home():
    return {
        "message": "Customer Behavior Prediction API is running",
        "model": "LightGBM",
        "threshold": 0.2,
    }