from fastapi import APIRouter

router = APIRouter(
    tags=["Health"],
)


@router.get("/health")
def health_check():
    """
    Check whether the API service is running.
    """
    return {
        "status": "healthy",
        "service": "Customer Behavior Prediction API",
    }