import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.lime_explainer import explain_customer_with_lime
from src.explainability.shap_explainer import explain_customer
from src.models.clv_predictor import predict_clv
from src.models.predictor import (
    load_model_and_config,
    predict_customer_response,
)
from src.segmentation.predictor import predict_customer_segment

st.set_page_config(
    page_title="Customer Behavior Prediction",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_prediction_resources():
    """
    Load the trained prediction model and configuration once.
    """
    return load_model_and_config()


model, config = load_prediction_resources()
@st.cache_data
def load_baseline_customer():
    """
    Create a baseline customer using median values from processed data.
    """
    data_path = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"

    data = pd.read_csv(data_path)

    model_features = config["features"]

    baseline = data[model_features].median(numeric_only=True)

    return baseline


st.title("Customer Behavior Prediction Dashboard")

st.write(
    "Predict customer campaign response and explore "
    "customer behavior insights."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Prediction Model",
        config.get("model", "LightGBM"),
    )

with col2:
    st.metric(
        "Decision Threshold",
        config.get("threshold", 0.2),
    )

with col3:
    st.metric(
        "Model Features",
        len(config["features"]),
    )

st.success("Prediction model loaded successfully.")

st.divider()

st.subheader("Customer Response Prediction")

st.write(
    "Enter the customer's basic information and purchasing behaviour "
    "to estimate their likelihood of responding to a campaign."
)

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0,
    step=1000.0,
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=45,
)

recency = st.number_input(
    "Recency (days since last purchase)",
    min_value=0,
    value=30,
)

total_spending = st.number_input(
    "Total Spending",
    min_value=0.0,
    value=500.0,
    step=50.0,
)

web_purchases = st.number_input(
    "Web Purchases",
    min_value=0,
    value=3,
)

catalog_purchases = st.number_input(
    "Catalog Purchases",
    min_value=0,
    value=2,
)

store_purchases = st.number_input(
    "Store Purchases",
    min_value=0,
    value=4,
)

if st.button("Predict Customer Response", type="primary"):

    baseline = load_baseline_customer()

    customer = baseline.copy()

    customer["Income"] = income
    customer["Age"] = age
    customer["Recency"] = recency
    customer["Total_Spending"] = total_spending
    customer["NumWebPurchases"] = web_purchases
    customer["NumCatalogPurchases"] = catalog_purchases
    customer["NumStorePurchases"] = store_purchases

    customer_df = pd.DataFrame([customer])

    result = predict_customer_response(customer_df)
    segment_result = predict_customer_segment(customer_df)

    segment_id = int(segment_result.iloc[0]["Segment_ID"])
    segment_name = segment_result.iloc[0]["Segment_Name"]
    clv_result = predict_clv(customer_df)

    predicted_clv = float(
    clv_result.iloc[0]["Predicted_CLV_Proxy"]
    )

    lime_explanation = explain_customer_with_lime(
    customer_df,
    num_features=5,
    )
    explanation = explain_customer(customer_df)

    top_explanation = explanation.head(5).copy()

    top_explanation["Impact"] = top_explanation["SHAP_Value"].apply(
        lambda value: (
        "Increases response likelihood"
        if value > 0
        else "Decreases response likelihood"
        )
    )

    probability = result.iloc[0]["response_probability"]
    prediction = result.iloc[0]["predicted_response"]
    label = result.iloc[0]["prediction_label"]

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    with result_col1:
        st.metric(
            "Response Probability",
            f"{probability:.1%}",
        )

    with result_col2:
        st.metric(
            "Prediction",
            label,
        )
    
    with result_col3:
        st.metric(
            "Customer Segment",
             segment_name,
        )
    with result_col4:
        st.metric(
        "Estimated Customer Value",
        f"{predicted_clv:,.2f}",
        )    

    st.caption(
        "Estimated Customer Value is a historical spending proxy "
        "based on Total Spending in the available dataset; "
        "it is not a forecast of future lifetime value."
    )

    if prediction == 1:
        st.success(
            "This customer is predicted to be likely to respond "
            "to the campaign."
        )
    else:
        st.info(
            "This customer is predicted to be unlikely to respond "
            "to the campaign."
        )

    st.divider()

    st.subheader("Why this prediction?")

    st.write(
        "The following features had the strongest influence "
        "on this customer's response prediction."
    )

    st.dataframe(
        top_explanation[
            [
                "Feature",
                "Feature_Value",
                "Impact",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "SHAP explains how each feature influenced the model's "
        "prediction. These effects describe model behaviour and "
        "should not be interpreted as causal relationships."
    )

    st.subheader("LIME Local Explanation")

    st.write(
        "LIME provides a second local explanation of this customer's "
        "prediction by approximating the model behaviour around this "
        "individual customer."
    )

    st.dataframe(
        lime_explanation[
            [
                "Feature_Rule",
                "LIME_Contribution",
                "Impact",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "LIME contributions show whether each local feature condition "
        "pushes the explanation toward or away from the positive "
        "'Likely to Respond' class. SHAP and LIME contribution "
        "magnitudes should not be compared directly."
    )    