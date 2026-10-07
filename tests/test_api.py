from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "Customer Behavior Prediction API"


def test_metadata_endpoint():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "LightGBM"
    assert data["threshold"] == 0.2
    assert data["feature_count"] == 38
    assert len(data["features"]) == 38

def test_predict_endpoint():
    payload = {
        "Income": 30000,
        "Kidhome": 1,
        "Teenhome": 1,
        "Recency": 60,
        "MntWines": 100,
        "MntFruits": 10,
        "MntMeatProducts": 30,
        "MntFishProducts": 5,
        "MntSweetProducts": 5,
        "MntGoldProds": 10,
        "NumDealsPurchases": 2,
        "NumWebPurchases": 2,
        "NumCatalogPurchases": 1,
        "NumStorePurchases": 3,
        "NumWebVisitsMonth": 7,
        "AcceptedCmp3": 0,
        "AcceptedCmp4": 0,
        "AcceptedCmp5": 0,
        "AcceptedCmp1": 0,
        "AcceptedCmp2": 0,
        "Complain": 0,
        "Z_CostContact": 3,
        "Z_Revenue": 11,
        "Total_Spending": 160,
        "Total_Children": 2,
        "Age": 50,
        "Customer_Tenure": 10,
        "Education_Basic": 0,
        "Education_Graduation": 1,
        "Education_Master": 0,
        "Education_PhD": 0,
        "Marital_Status_Alone": 0,
        "Marital_Status_Divorced": 0,
        "Marital_Status_Married": 1,
        "Marital_Status_Single": 0,
        "Marital_Status_Together": 0,
        "Marital_Status_Widow": 0,
        "Marital_Status_YOLO": 0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "response_probability" in data
    assert "predicted_response" in data
    assert "prediction_label" in data
    assert "threshold_used" in data

    assert 0 <= data["response_probability"] <= 1
    assert data["predicted_response"] in [0, 1]
    assert data["threshold_used"] == 0.2

def test_batch_predict_endpoint():
    csv_content = (
        "Income,Kidhome,Teenhome,Recency,MntWines,MntFruits,"
        "MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,"
        "NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,"
        "NumStorePurchases,NumWebVisitsMonth,AcceptedCmp3,AcceptedCmp4,"
        "AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Complain,Z_CostContact,"
        "Z_Revenue,Total_Spending,Total_Children,Age,Customer_Tenure,"
        "Education_Basic,Education_Graduation,Education_Master,"
        "Education_PhD,Marital_Status_Alone,Marital_Status_Divorced,"
        "Marital_Status_Married,Marital_Status_Single,"
        "Marital_Status_Together,Marital_Status_Widow,"
        "Marital_Status_YOLO\n"
        "30000,1,1,60,100,10,30,5,5,10,2,2,1,3,7,"
        "0,0,0,0,0,0,3,11,160,2,50,10,"
        "0,1,0,0,0,0,1,0,0,0,0\n"
    )

    files = {
        "file": (
            "customers.csv",
            csv_content,
            "text/csv",
        )
    }

    response = client.post(
        "/predict/batch",
        files=files,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")

    response_text = response.text

    assert "response_probability" in response_text
    assert "predicted_response" in response_text
    assert "prediction_label" in response_text