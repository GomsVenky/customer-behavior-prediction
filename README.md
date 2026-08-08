# Customer Behavior Prediction using Machine Learning

## 📌 Project Overview

This project aims to analyze customer purchasing behavior and build a machine learning model to predict whether a customer will respond to a marketing campaign.

The project covers:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Preprocessing
- Model Training
- Model Evaluation
- Business Insights

## 📂 Dataset

**Dataset:** Marketing Campaign Dataset

**Number of records:** 2240

**Number of features:** 29 (before feature engineering)

**Target Variable:**

`Response`

- `1` → Customer accepted the campaign
- `0` → Customer did not accept the campaign

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## ✅ Project Progress

### ✔ Phase 1 — Data Loading

Loaded dataset using Pandas.

Verified data types.

Explored dataset structure.

Used:

- `head()`
- `info()`
- `describe()`

### ✔ Phase 2 — Data Cleaning

Completed:

- Checked missing values
- Filled missing values in `Income` using Median
- Checked duplicate records
- Verified data types
- Detected outliers using IQR and Boxplots
- Identified unrealistic ages above 100 years
- Removed unrealistic age records

### ✔ Phase 3 — Exploratory Data Analysis (Completed)

#### Univariate Analysis

Analyzed:

- Income Distribution
- Education
- Marital Status
- Age
- Customer Response

Techniques:

- Histogram
- Countplot
- Boxplot
- Summary Statistics

#### Bivariate Analysis

Studied relationships between:

- Income vs Total Spending
- Education vs Total Spending
- Marital Status vs Total Spending
- Total Children vs Total Spending
- Age vs Income
- Recency vs Total Spending
- Recency vs Response
- Income vs Response

Visualizations Used:

- Scatter Plot
- Box Plot
- Correlation Heatmap

### ✔ Phase 4 — Feature Engineering

Created new features to improve model performance and business understanding:

- Created `Total_Spending` by combining spending across all product categories.
- Created `Total_Children` by combining `Kidhome` and `Teenhome`.
- Created `Age` from `Year_Birth` and removed unrealistic age records (>100 years).
- Created `Customer_Tenure` from `Dt_Customer`.
- Created `Total_Purchase` from web, catalog, and store purchases.
- Simplified `Marital_Status` into broader categories where appropriate.
- Dropped redundant columns (`Year_Birth` and `Dt_Customer`) after creating the new features.

### ✔ Phase 5 — Machine Learning Preprocessing

Performed preprocessing steps before model training:

- Applied One-Hot Encoding to categorical variables.
- Used `drop_first=True` to remove redundant dummy variables.
- Verified all features were numeric.
- Separated features (`X`) and target (`y`).
- Split the dataset into training (80%) and testing (20%) sets using `train_test_split()`.
- Applied `StandardScaler` to scale numerical features.
- Used `fit_transform()` on the training data.
- Used `transform()` on the testing data to prevent data leakage.

Final dataset split:

- Training set: 1789 records
- Testing set: 448 records
- Features after encoding: 39

### ✔ Phase 7 —  Model Evaluation
Accuracy

The Logistic Regression model achieved an accuracy of:

0.88839 (88.84%)

This means approximately 88.84% of the predictions made on the testing dataset matched the actual responses.

Confusion Matrix

The confusion matrix obtained was:

[[364  12]
 [ 38  34]]

 Interpretation:

 | Metric              | Value | Meaning                                                                      |
| ------------------- | ----: | ---------------------------------------------------------------------------- |
| True Negative (TN)  |   364 | Customers who did not respond and were correctly predicted as non-responders |
| False Positive (FP) |    12 | Customers who did not respond but were incorrectly predicted as responders   |
| False Negative (FN) |    38 | Customers who responded but were incorrectly predicted as non-responders     |
| True Positive (TP)  |    34 | Customers who responded and were correctly predicted as responders           |

The confusion matrix shows that although the model achieved relatively high overall accuracy, it missed a considerable number of actual responders. Therefore, accuracy alone may not be sufficient for evaluating this imbalanced classification problem.

🎯 Key Insights
Higher income customers generally spend more.
Customers with Basic education tend to spend less than other education groups.
Customers without children show higher median spending.
Rare marital status categories (YOLO, Absurd) contain very few records and should be interpreted carefully.
The Response variable is highly imbalanced (~85% No Response vs ~15% Response).
The model achieved 88.84% accuracy, but the confusion matrix indicates that several actual responders were missed.

🚀 Upcoming Work

### Model Evaluation
Precision
Recall
F1-score
ROC-AUC
Detailed evaluation of the imbalanced target variable

### Model Development
Train additional classification models
Compare model performance
Perform hyperparameter tuning
Select the most suitable final model

### Business Recommendations
Identify customer groups most likely to respond
Develop marketing targeting strategies
Analyze the trade-off between false positives and false negatives
Provide final business recommendations based on model results
