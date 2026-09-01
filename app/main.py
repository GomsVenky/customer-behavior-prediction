from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd


# Create FastAPI application
app = FastAPI(
    title="Customer Behavior Prediction API",
    description="API for predicting customer campaign response using LightGBM.",
    version="1.0.0"
)


# Load trained model
model = joblib.load("models/final_lightgbm_model.pkl")

# Load model configuration
config = joblib.load("models/model_config.pkl")

FEATURES = config["features"]
THRESHOLD = config["threshold"]


class CustomerData(BaseModel):
    Income: float
    Kidhome: int
    Teenhome: int
    Recency: int
    MntWines: int
    MntFruits: int
    MntMeatProducts: int
    MntFishProducts: int
    MntSweetProducts: int
    MntGoldProds: int
    NumDealsPurchases: int
    NumWebPurchases: int
    NumCatalogPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int
    AcceptedCmp3: int
    AcceptedCmp4: int
    AcceptedCmp5: int
    AcceptedCmp1: int
    AcceptedCmp2: int
    Complain: int
    Z_CostContact: float
    Z_Revenue: float
    Total_Spending: int
    Total_Children: int
    Age: int
    Customer_Tenure: int
    Education_Basic: int
    Education_Graduation: int
    Education_Master: int
    Education_PhD: int
    Marital_Status_Alone: int
    Marital_Status_Divorced: int
    Marital_Status_Married: int
    Marital_Status_Single: int
    Marital_Status_Together: int
    Marital_Status_Widow: int
    Marital_Status_YOLO: int


@app.get("/")
def home():
    return {
        "message": "Customer Behavior Prediction API is running",
        "model": "LightGBM",
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(customer: CustomerData):

    try:
        # Convert input to DataFrame
        customer_data = pd.DataFrame([customer.model_dump()])

        # Ensure correct feature order
        customer_data = customer_data[FEATURES]

        # Generate probability
        probability = model.predict_proba(customer_data)[:, 1][0]

        # Apply selected threshold
        prediction = int(probability >= THRESHOLD)

        label = (
            "Likely to Respond"
            if prediction == 1
            else "Not Likely to Respond"
        )

        return {
            "response_probability": float(probability),
            "predicted_response": prediction,
            "prediction_label": label,
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )