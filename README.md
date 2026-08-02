Customer Behavior Prediction using Machine Learning
📌 Project Overview

This project aims to analyze customer purchasing behavior and build a machine learning model to predict whether a customer will respond to a marketing campaign.

The project covers:

Data Cleaning
Exploratory Data Analysis (EDA)
Feature Engineering
Machine Learning
Model Evaluation
Business Insights
📂 Dataset

Dataset: Marketing Campaign Dataset

Number of records: 2240

Number of features: 29 (before feature engineering)

Target Variable:

Response
1 → Customer accepted the campaign
0 → Customer did not accept the campaign
🛠 Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn (Coming soon)
✅ Project Progress
✔ Phase 1 — Data Loading
Loaded dataset using Pandas
Verified data types
Explored dataset structure
Used:
head()
info()
describe()
✔ Phase 2 — Data Cleaning

Completed:

Checked missing values
Filled missing values in Income using Median
Checked duplicate records
Verified data types
Detected outliers using IQR and Boxplots
✔ Phase 3 — Exploratory Data Analysis (Completed)
Univariate Analysis

Analyzed:

Income Distribution
Education
Marital Status

Techniques:

Histogram
Countplot
Boxplot
Summary Statistics
Bivariate Analysis

Studied relationships between:

Income vs Total Spending
Education vs Total Spending
Marital Status vs Total Spending
Total Children vs Total Spending

Visualizations Used:

Scatter Plot
Box Plot
Correlation Heatmap
✔ Feature Engineering

Created new features to improve model performance and business understanding:

- Created `Total_Spending` by combining spending across all product categories.
- Created `Total_Children` by combining `Kidhome` and `Teenhome`.
- Created `Age` from `Year_Birth` and removed unrealistic age records (>100 years).
- Created `Customer_Tenure` from `Dt_Customer`.
- Dropped redundant columns (`Year_Birth` and `Dt_Customer`) after creating the new features.

## Machine Learning Preprocessing

Performed preprocessing steps before model training:

- Applied One-Hot Encoding to categorical variables.
- Removed redundant dummy variables using `drop_first=True`.
- Verified all features were numeric.
- Separated features (`X`) and target (`y`).
- Split the dataset into training (80%) and testing (20%) sets using `train_test_split()`.

🎯 Key Insights

You can write something like:

Higher income customers generally spend more.
Customers with Basic education tend to spend less than other education groups.
Customers without children show higher median spending.
Rare marital status categories (YOLO, Absurd) contain very few records and should be interpreted carefully.
The Response variable is highly imbalanced (~85% No Response vs ~15% Response).

(We'll keep adding insights as we progress.)

🚀 Upcoming Work
Data Preprocessing
Model Training
Model Evaluation
Business Recommendations