from fastapi import APIRouter

from src.models.predictor import load_model_and_config

router = APIRouter(
    tags=["Metadata"],
)


_, config = load_model_and_config()


@router.get("/metadata")
def get_model_metadata():
    """
    Return metadata about the deployed prediction model.
    """
    return {
        "model": config.get("model", "LightGBM"),
        "threshold": config.get("threshold", 0.2),
        "features": config["features"],
        "feature_count": len(config["features"]),
    }