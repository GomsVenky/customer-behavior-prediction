# Customer Behavior Prediction

An end-to-end Machine Learning Engineering project for customer behaviour analysis, including campaign response prediction, customer segmentation, historical customer-value estimation, explainable AI, deep-learning experiments, REST API deployment, an interactive dashboard, PostgreSQL persistence, automated testing, and Docker-based deployment.

## Project Overview

The project analyses customer demographic, purchasing, engagement, and campaign data to support three main business tasks:

1. Predict whether a customer is likely to respond to a marketing campaign.
2. Segment customers into meaningful behavioural/value groups.
3. Estimate historical customer value using a spending-based proxy.

The complete solution extends beyond model experimentation and includes reusable Python modules, FastAPI, Streamlit, PostgreSQL, Docker Compose, automated tests, structured logging, and project documentation.

## Dataset

The project uses `marketing_campaign.csv`.

The original dataset contains:

- **2,240 customer records**
- **29 columns**
- Demographic information
- Household information
- Product-category spending
- Purchase-channel activity
- Previous campaign interactions
- Campaign response

After cleaning, the final dataset contains **2,237 customers**.

The campaign-response target is imbalanced:

- **85.07%** non-responders
- **14.93%** responders

Because of this imbalance, model evaluation uses Precision, Recall, F1-score, and ROC-AUC in addition to Accuracy.

## Data Cleaning and Feature Engineering

The reusable data pipeline performs:

- Raw data loading
- Missing-value handling
- Data-quality validation
- Age-outlier removal
- Feature engineering
- Categorical encoding
- Processed-data generation

`Income` contains 24 missing values and is imputed using the median. Three unrealistic age records above 100 years are removed.

Important engineered features include:

| Feature | Definition |
| --- | --- |
| `Total_Spending` | Sum of the six product-category spending variables |
| `Total_Children` | `Kidhome + Teenhome` |
| `Age` | `2026 - Year_Birth` |
| `Customer_Tenure` | `2026 - year(Dt_Customer)` |

`Education` and `Marital_Status` are one-hot encoded.

Customer `ID`, `Dt_Customer`, and `Year_Birth` are excluded from the final response-model inputs after the required derived features are created.

The final response model uses **38 features**.

### Run the Data Pipeline

From the project root:

```bash
python -m src.data.pipeline
```

The processed dataset is generated at:

```text
data/processed/customer_features.csv
```

The verified processed output contains **2,237 rows, 39 columns including the target, and no missing values**.

## Campaign Response Prediction

Multiple classical machine-learning models were evaluated.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 87.95% | 65.12% | 41.79% | 50.91% | 88.04% |
| Decision Tree | 88.17% | 73.33% | 32.84% | 45.36% | 72.91% |
| Random Forest | 89.06% | 82.14% | 34.33% | 48.42% | 90.35% |
| KNN | 87.95% | 65.85% | 40.30% | 50.00% | 78.07% |
| SVM | 88.62% | 75.00% | 35.82% | 48.48% | 88.43% |
| LightGBM - threshold 0.50 | 89.06% | 70.45% | 46.27% | 55.86% | **91.04%** |
| LightGBM - threshold 0.20 | 87.95% | 60.00% | **58.21%** | **59.09%** | **91.04%** |

### Production Model

**LightGBM** was selected as the production response-prediction model.

The deployed classification threshold is:

```text
0.20
```

The lower threshold improves responder recall compared with the default 0.50 threshold, which is useful when the business objective is to identify more potential campaign responders.

The production model and configuration are stored in:

```text
models/final_lightgbm_model.pkl
models/model_config.pkl
```

The configuration preserves the exact feature order and deployed decision threshold.

## Customer Segmentation

Customer segmentation was performed using behavioural and demographic features including:

- Income
- Total Spending
- Total Children
- Age
- Recency
- Web Purchases
- Catalog Purchases
- Store Purchases

### K-Means

K-Means with **2 clusters** was selected as the primary segmentation model.

| Method | Silhouette Score | Davies-Bouldin Index |
| --- | ---: | ---: |
| **K-Means** | **0.3275** | **1.2728** |
| Gaussian Mixture Model | 0.3091 | 1.3110 |

K-Means provided slightly better separation according to both evaluation metrics.

The two resulting groups can be interpreted as:

- **Lower-Value Customers** — lower average income, spending, and purchase activity.
- **High-Value Customers** — higher average income, spending, and purchase activity.

The observed campaign response rate was approximately **9.62%** for the lower-value cluster and **21.37%** for the higher-value cluster.

This is an observed association and should not be interpreted as a causal relationship.

## Customer Value Estimation

The project includes a regression model for estimating customer value.

The dataset does not contain a true future Customer Lifetime Value target. Therefore:

```text
CLV_Proxy = Total_Spending
```

is used as a **historical customer-value proxy**.

It must not be interpreted as a prediction of true future CLV.

The predictors are:

- `Income`
- `NumWebPurchases`
- `NumCatalogPurchases`
- `NumStorePurchases`
- `Customer_Tenure`

`CLV_Proxy` is excluded from the predictor variables to prevent target leakage.

### Regression Results

| Model | MAE | RMSE | R-Squared |
| --- | ---: | ---: | ---: |
| Linear Regression | 198.28 | 298.69 | 0.7651 |
| **Random Forest Regression** | **109.57** | **208.75** | **0.8853** |

Random Forest Regression was selected as the final customer-value model.

The saved artifacts are:

```text
models/final_clv_rf_model.pkl
models/clv_model_config.pkl
```

Application outputs deliberately use the name `Predicted_CLV_Proxy` so the historical-proxy limitation remains visible.

## Explainable AI

The project uses three complementary explainability techniques:

### SHAP

SHAP provides global and customer-level explanations of the LightGBM response model.

Individual explanations were evaluated for representative:

- Low-probability customers
- High-probability customers
- Borderline customers near the deployed `0.20` threshold

The Streamlit application displays the **top five SHAP feature contributions** for the entered customer profile.

### LIME

LIME provides a second local explanation of individual predictions using interpretable feature rules.

The dashboard displays the **top five LIME contributions** and indicates whether each contribution increases or decreases support for the positive response class.

SHAP and LIME values should not be compared directly by magnitude because they use different explanation frameworks.

### Partial Dependence

Partial Dependence analysis was performed on selected important features.

Observed model behaviour included:

- Increasing `Recency` generally reduced predicted response.
- `Total_Spending` showed a nonlinear relationship with predicted response.
- `NumWebVisitsMonth` showed increasing predicted response up to approximately nine visits followed by a relative plateau.

These patterns describe the fitted model and **must not be interpreted as causal effects**.

## Deep Learning Experiments

Deep-learning models were explored as comparative and learning experiments. Because this is a relatively small tabular dataset, the final production response model remains LightGBM.

### Artificial Neural Network

The ANN uses the same 38 response-prediction features, with feature scaling fitted only on the training data.

Architecture:

```text
Input (38 features)
    ↓
Dense (64, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.30)
    ↓
Dense (32, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.30)
    ↓
Dense (16, ReLU)
    ↓
Dense (1, Sigmoid)
```

Early stopping was used during training.

The validation set was used to select a classification threshold of **0.32**, after which the model was evaluated on the untouched test set.

| Evaluation | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.50 | 90.77% | 78.79% | 52.00% | 62.65% | 90.85% |
| Validation | 0.32 | 90.18% | 64.91% | 74.00% | 69.16% | 90.85% |
| **Final Test** | **0.32** | **88.17%** | **60.61%** | **59.70%** | **60.15%** | **88.38%** |

LightGBM remains the production classifier because it achieved stronger test ROC-AUC and provides a simpler deployment path for this tabular dataset.

### Autoencoder Anomaly Detection

An autoencoder was trained using ten customer spending and purchase-behaviour features.

Architecture:

```text
10 → 8 → 4 → 8 → 10
```

The anomaly threshold was defined as the **95th percentile of training reconstruction error**.

| Dataset | Anomalies | Records | Rate |
| --- | ---: | ---: | ---: |
| Training | 90 | 1,789 | 5.03% |
| Test | 26 | 448 | 5.80% |

The dataset does not contain ground-truth anomaly labels. Therefore, these records represent customers with unusually high reconstruction error and are **not confirmed anomalies**.

### LSTM Feasibility Limitation

An LSTM was considered for sequential customer-purchase modelling.

However, the available dataset contains aggregated customer records rather than timestamped purchase-event sequences. `Dt_Customer` represents customer enrolment and cannot be treated as a sequence of customer purchases.

Creating artificial sequences from the aggregated features would fabricate temporal information and produce a methodologically invalid experiment.

Therefore, an LSTM was **deliberately not trained**.

A valid future LSTM experiment would require transaction-level data containing at minimum:

- Customer identifier
- Transaction timestamp
- Purchase amount and/or product category

This limitation is documented rather than generating artificial sequential data merely to satisfy a model requirement.

## FastAPI REST API

The production LightGBM model is exposed through a FastAPI application.

The API includes:

- Health checking
- Model metadata
- Single-customer prediction
- Batch CSV prediction
- Pydantic request validation
- PostgreSQL prediction persistence
- Structured JSON logging

### Run the API Locally

From the project root:

```bash
uvicorn api.main:app --reload
```

The API runs on port `8000`.

Interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Prediction Output

A single response prediction returns:

- Response probability
- Predicted response (`0` or `1`)
- Business-readable prediction label
- Classification threshold used

The deployed threshold is **0.20**.

## PostgreSQL Persistence

PostgreSQL is used to persist application results through SQLAlchemy.

The database stores:

- Response predictions
- Customer segment assignments

Database tables are initialized automatically when the Dockerized API starts.

The relationship between prediction and segmentation records through `customer_id` is conceptual; the current database schema does not impose a foreign-key constraint between the two tables.

## Streamlit Dashboard

The interactive Streamlit dashboard combines the major project outputs into a single customer-level view.

It displays:

- Campaign response probability
- Response prediction label
- Customer segment
- Historical CLV proxy estimate
- Top five SHAP contributions
- Top five LIME contributions

Run it locally with:

```bash
streamlit run frontend/dashboard.py
```

The dashboard runs on port `8501`.

### Dashboard Input Limitation

For usability, seven important customer attributes are directly editable in the demonstration dashboard:

- Income
- Age
- Recency
- Total Spending
- Web Purchases
- Catalog Purchases
- Store Purchases

The remaining production-model features are populated using median values from the processed dataset.

The dashboard should therefore be treated as a project demonstration interface rather than a complete production customer-data entry system.

## Docker Deployment

The complete application is containerized using Docker Compose.

The stack contains:

| Service | Purpose | Port |
| --- | --- | ---: |
| API | FastAPI prediction service | 8000 |
| Dashboard | Streamlit interface | 8501 |
| Database | PostgreSQL | 5432 |

### Environment Configuration

Create the local environment file from the provided example:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env` or real credentials to source control.

### Start the Full Stack

Ensure Docker Desktop is running, then execute:

```bash
docker compose up -d --build
```

Check the containers with:

```bash
docker compose ps
```

The expected services are:

```text
api
dashboard
db
```

The PostgreSQL service includes a health check, and the API waits for the database to become healthy before starting.

### Stop the Stack

```bash
docker compose down
```

The complete Dockerized workflow has been validated successfully, including database initialization, API health checking, Swagger prediction, PostgreSQL persistence, and Streamlit dashboard access.

## Testing and Code Quality

The project includes automated unit and API tests covering the main production components.

Run the complete test suite with coverage:

```bash
pytest --cov=src --cov=api --cov-report=term-missing
```

Final verified result:

```text
26 passed
Total coverage: 76%
```

The tests include explicit leakage checks verifying that:

- `Response` is not included in prediction features.
- Customer `ID` is not included in prediction features.
- `Year_Birth` and `Dt_Customer` are removed after derived features are created.
- Engineered `Age` and `Customer_Tenure` are retained.

### Static Analysis

Run Ruff with:

```bash
ruff check src api frontend tests
```

Final verified result:

```text
All checks passed!
```

## Reproducibility

The project uses several safeguards to improve reproducibility:

- Fixed random seeds for model experiments
- YAML-based project configuration
- Saved model configurations and feature order
- `scikit-learn==1.9.0` pinned to match the serialized production artifacts
- Environment variables for deployment configuration
- Reusable raw-to-processed data pipeline
- Docker-based application deployment
- Automated tests and leakage checks

Project configuration is stored in:

```text
config/config.yaml
```

An example environment configuration is provided in:

```text
.env.example
```

Structured JSON logging is implemented for prediction activity without recording the complete customer input payload.

## Documentation

Detailed project documentation is available under `docs/`.

```text
docs/
├── architecture_diagram.png
├── er_diagram.png
├── final_report.pdf
├── create_architecture_diagram.py
├── create_er_diagram.py
└── create_final_report.py
```

The final report covers:

- Problem statement and dataset
- Data preparation and feature engineering
- Classical model comparison
- LightGBM model selection and threshold tuning
- Customer segmentation
- Historical CLV proxy modelling
- SHAP, LIME, and PDP explainability
- ANN and autoencoder experiments
- LSTM feasibility limitation
- FastAPI and PostgreSQL
- Streamlit dashboard
- Docker deployment
- Testing and reproducibility
- Business recommendations and project limitations

## Project Structure

```text
customer-behavior-prediction/
│
├── api/
│   ├── main.py
│   ├── routers/
│   └── schemas/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   ├── architecture_diagram.png
│   ├── er_diagram.png
│   ├── final_report.pdf
│   └── report/diagram generation scripts
│
├── frontend/
│   └── dashboard.py
│
├── models/
│   ├── final_lightgbm_model.pkl
│   ├── model_config.pkl
│   ├── final_clv_rf_model.pkl
│   └── clv_model_config.pkl
│
├── models_artifacts/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   ├── explainability/
│   ├── features/
│   ├── models/
│   ├── segmentation/
│   └── utils/
│
├── tests/
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

### Notebook Organization

The experimental workflow is intentionally retained in one comprehensive notebook:

```text
notebooks/01_eda.ipynb
```

It contains EDA, feature engineering, classical model experiments, boosting models, segmentation, customer-value modelling, explainability, ANN, autoencoder analysis, and the LSTM feasibility assessment.

Although the original project specification proposed separate EDA, feature-engineering, and model-experiment notebooks, the work was consolidated into one notebook while production functionality was separated into reusable Python modules.

## Key Limitations

- The dataset is relatively small and contains aggregated customer records.
- The CLV target is a historical spending proxy, not true future Customer Lifetime Value.
- Transaction-level sequential data is unavailable, so a valid LSTM could not be trained.
- The autoencoder does not have ground-truth anomaly labels.
- The Streamlit dashboard directly exposes only seven primary customer inputs.
- Model explanations and segment relationships represent learned associations, not causal effects.
- Model performance should be monitored and re-evaluated when applied to new customer populations.

## Business Use

The system can support marketing teams by:

- Prioritizing customers with higher campaign-response probability.
- Identifying higher-value customer segments for targeted analysis.
- Providing a historical customer-value estimate.
- Explaining individual predictions using SHAP and LIME.
- Supporting batch scoring through the FastAPI service.

The model should be used as a **decision-support tool**, not as a replacement for business judgement.

## Conclusion

This project demonstrates an end-to-end machine learning engineering workflow, progressing from raw customer data through data validation, feature engineering, model development, segmentation, customer-value estimation, explainable AI, deep-learning experiments, testing, API deployment, database persistence, dashboard development, and containerization.

LightGBM was selected as the production campaign-response model with a deployed threshold of **0.20**, achieving **91.04% ROC-AUC** and **58.21% recall** on the held-out test set.

K-Means provides business-readable customer segmentation, while Random Forest Regression provides the historical customer-value proxy. SHAP and LIME provide customer-level prediction explanations.

The final solution is deployed as an integrated **FastAPI + Streamlit + PostgreSQL** application orchestrated with **Docker Compose**, with automated testing, leakage checks, structured logging, configuration management, and reproducible data processing.

Where the available dataset could not support a methodologically valid requirement—particularly LSTM sequence modelling and true future CLV forecasting—the limitation was explicitly documented rather than introducing fabricated data.