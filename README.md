# Customer Behavior Prediction

## 📌 Project Overview

This project uses machine learning to predict whether a customer is likely to respond to a marketing campaign.

The project focuses on understanding customer behaviour through exploratory data analysis, feature engineering, preprocessing, classification model development, cross-validation, hyperparameter tuning, and final model evaluation.

The target variable is `Response`, where:

* `0` = Customer did not respond
* `1` = Customer responded

The target variable is imbalanced, with approximately **85.07% non-responders** and **14.93% responders**. Therefore, model evaluation considers Precision, Recall, F1-score and ROC-AUC in addition to Accuracy.

---

## 🎯 Project Objective

The main objective is to build a classification model that can identify customers who are likely to respond to a marketing campaign.

The model can help marketing teams prioritize potential responders and support more targeted customer campaigns.

---

## 📊 Dataset

The project uses the `marketing_campaign.csv` dataset.

The original dataset contains **2,240 records and 29 columns**.

During data cleaning, missing values and data-quality issues were investigated. After cleaning, the modelling dataset contained **2,237 customer records**.

The dataset contains customer demographic information, purchasing behaviour, website activity, previous campaign responses, and the campaign response target.

---

## 🧹 Data Cleaning

The following data-cleaning activities were performed:

* Inspected the dataset structure and data types.
* Checked for missing values.
* Identified missing values in `Income`.
* Investigated potential data-quality issues and unusual values.
* Checked for duplicate records.
* Performed basic data-quality checks.

---

## 🔧 Feature Engineering

Several new features were created to better represent customer behaviour:

### Age

Created from `Year_Birth`.

### Total_Spending

Created by combining spending across:

* `MntWines`
* `MntFruits`
* `MntMeatProducts`
* `MntFishProducts`
* `MntSweetProducts`
* `MntGoldProds`

### Total_Children

Created by combining:

* `Kidhome`
* `Teenhome`

### Customer_Tenure

Created from `Dt_Customer`.

The original product-category spending variables were retained because they may contain additional predictive information.

After feature engineering, `Year_Birth` and `Dt_Customer` were removed before machine-learning preprocessing.

---

## 🔄 Machine Learning Preprocessing

Categorical variables were converted into numerical variables using one-hot encoding.

The following categorical variables were encoded:

* `Education`
* `Marital_Status`

`drop_first=True` was used to reduce redundant dummy variables.

The final modelling dataset initially contained 40 columns after encoding, including the target and customer identifier.

### Customer ID Removal

During feature-importance analysis, the `ID` feature showed relatively high importance in the initial Random Forest model.

Since `ID` is an identifier rather than a meaningful customer behavioural or demographic feature, it was removed from the final feature set.

All five models were subsequently re-evaluated using the corrected **38-feature dataset** with the same final train-test split and test customers.

---

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics and relationships within the dataset.

The analysis included:

* Univariate analysis
* Bivariate analysis
* Scatter plots
* Box plots
* Bar plots
* Distribution analysis
* Customer spending analysis
* Income vs spending analysis
* Response distribution analysis

The response distribution showed a clear class imbalance, which was considered when selecting evaluation metrics.

---

## 🤖 Machine Learning Models

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

## 🔁 Cross-Validation

Five-fold Stratified Cross-Validation was used to evaluate model stability while preserving the class distribution across folds.

`StratifiedKFold` was configured with:

* `n_splits = 5`
* `shuffle = True`
* `random_state = 0`

Cross-validation was used before hyperparameter tuning to compare model stability across multiple validation folds.

---

## ⚙️ Hyperparameter Tuning

`GridSearchCV` with five-fold Stratified Cross-Validation was used for hyperparameter tuning.

The tuning objective used **F1-score**, because the target variable is imbalanced and accuracy alone does not adequately represent the model's ability to identify responders.

### Logistic Regression

Best parameter:

```text
C = 10
```

### Decision Tree

Best parameters:

```text
criterion = gini
max_depth = 7
min_samples_leaf = 1
min_samples_split = 5
```

### Random Forest

Best parameters:

```text
n_estimators = 200
max_depth = None
max_features = sqrt
min_samples_leaf = 1
min_samples_split = 2
```

### KNN

Best parameters:

```text
metric = euclidean
n_neighbors = 3
weights = distance
```

### SVM

Best parameters:

```text
C = 0.1
gamma = scale
kernel = linear
```

The SVM probability outputs were obtained using `CalibratedClassifierCV`.

---

## 🏆 Final Model Evaluation

After identifying the customer `ID` as an inappropriate predictive feature, all five models were re-evaluated using the corrected 38-feature dataset.

The final evaluation used an **80:20 train-test split**, with stratification and `random_state=0`.

### Final Test Results

| Model               |   Accuracy |  Precision |     Recall |   F1-score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     87.95% |     65.12% | **41.79%** | **50.91%** |     88.04% |
| Decision Tree       |     88.17% |     73.33% |     32.84% |     45.36% |     72.91% |
| **Random Forest**   | **89.06%** | **82.14%** |     34.33% |     48.42% | **90.35%** |
| KNN                 |     87.95% |     65.85% |     40.30% |     50.00% |     78.07% |
| SVM                 |     88.62% |     75.00% |     35.82% |     48.48% |     88.43% |

---

## 🥇 Final Model Selection — Random Forest

**Random Forest was selected as the final model.**

It achieved:

* **89.06% Accuracy**
* **82.14% Precision**
* **34.33% Recall**
* **48.42% F1-score**
* **90.35% ROC-AUC**

Random Forest achieved the highest Accuracy, Precision and ROC-AUC among the five evaluated models.

Precision is particularly important for this business problem because the objective is to identify customers who are genuinely likely to respond while reducing unnecessary marketing efforts.

Logistic Regression achieved higher Recall and F1-score, but Random Forest provided substantially higher Precision and the highest ROC-AUC.

Therefore, Random Forest provides a stronger choice for targeted customer selection under the current business objective.

---

## 🔍 Feature Importance

Feature-importance analysis was performed using the final Random Forest model.

The top predictive features were:

| Rank | Feature           | Importance |
| ---: | ----------------- | ---------: |
|    1 | Recency           |     0.0936 |
|    2 | Total_Spending    |     0.0721 |
|    3 | MntMeatProducts   |     0.0645 |
|    4 | Income            |     0.0626 |
|    5 | MntWines          |     0.0614 |
|    6 | MntGoldProds      |     0.0523 |
|    7 | Age               |     0.0462 |
|    8 | MntSweetProducts  |     0.0432 |
|    9 | NumWebVisitsMonth |     0.0425 |
|   10 | AcceptedCmp5      |     0.0388 |

### Key Observation

`Recency` was the most important feature in the final Random Forest model, indicating that recent customer activity provides a strong predictive signal for campaign response.

Purchasing behaviour, including `Total_Spending` and individual product-category spending, was also highly influential.

`Income`, `Age`, and `NumWebVisitsMonth` also provided meaningful predictive information.

Feature importance indicates predictive contribution and should not be interpreted as a causal relationship.

---

## 💼 Business Recommendations

Based on the final Random Forest model and feature-importance analysis:

### 1. Prioritize recently active customers

Recency was the most important predictive feature.

Customers with more recent purchasing activity can be prioritized for targeted marketing campaigns.

### 2. Consider customer spending behaviour

Total spending and several product-category spending variables were among the most important features.

Customers with stronger purchasing activity can be considered higher-priority campaign targets.

### 3. Use digital engagement as a targeting signal

`NumWebVisitsMonth` ranked among the important features.

Website engagement may therefore provide a useful signal when identifying potential campaign responders.

### 4. Consider previous campaign engagement

`AcceptedCmp5` and `AcceptedCmp3` were among the important predictive features.

Previous campaign engagement can help identify customers who may be more likely to respond to future campaigns.

### 5. Use the model as a decision-support tool

The Random Forest achieved **82.14% precision** and **90.35% ROC-AUC** on the unseen test data.

The model can therefore support marketing teams in prioritizing customers rather than replacing business judgement.

---

## ⚠️ Model Limitation

The final Random Forest recall was **34.33%**.

This means that some customers who would respond to the campaign may not be identified by the current classification threshold.

The model therefore prioritizes reliable positive predictions over identifying every possible responder.

---

## 🚀 Future Improvements

Potential future improvements include:

* Probability-threshold optimization to improve the Precision-Recall trade-off.
* Further analysis of false positives and false negatives.
* Additional feature selection and feature engineering.
* Testing alternative class-imbalance handling techniques.
* Evaluating whether calibrated probabilities can improve campaign targeting.
* Exploring additional model evaluation techniques before deployment.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📁 Project Structure

```text
customer-behavior-prediction/
│
├── data/
│   └── marketing_campaign.csv
│
├── models/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── outputs/
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📌 Conclusion

This project developed and evaluated multiple machine-learning classification models for predicting customer response to marketing campaigns.

The workflow included data cleaning, exploratory data analysis, feature engineering, categorical encoding, feature scaling where appropriate, cross-validation, hyperparameter tuning, feature-importance analysis, and final evaluation on unseen test data.

After identifying and removing the customer `ID` feature, all five models were re-evaluated using a consistent 38-feature dataset.

Among the evaluated models, **Random Forest was selected as the final model**, achieving **89.06% accuracy, 82.14% precision, and 90.35% ROC-AUC**.

The results indicate that recent customer activity, spending behaviour, income, age, online engagement, and previous campaign responses provide useful predictive signals for identifying potential marketing campaign responders.
