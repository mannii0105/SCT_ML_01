🏠 House Price Prediction - Kaggle Ames Housing Dataset
This repository contains my solution for Task 01 of my Machine Learning Internship at SkillCraft Technology. The goal is to implement a linear regression model to predict house prices based on physical features like square footage, bedrooms, and bathrooms.

📋 Project Overview
Predicting house prices is a classic regression problem. Using the Ames Housing Dataset, which contains 79 explanatory variables, this project focuses on identifying the most influential factors that drive real estate value.

Key Objectives:
Perform Exploratory Data Analysis (EDA) to understand data distributions.

Handle missing values and preprocess data for machine learning.

Implement a Multiple Linear Regression model.

Evaluate model performance using Root Mean Squared Error (RMSE) on log-transformed prices.

🛠️ Tech Stack
Language: Python 3.x

Libraries: * pandas for data manipulation

numpy for numerical operations

scikit-learn for machine learning modeling

matplotlib & seaborn for data visualization

📊 Dataset Description
The dataset includes 1460 training observations and 1459 test observations. While the dataset offers 79 features, our primary focus for Task 01 includes:

GrLivArea: Above grade (ground) living area square feet.

BedroomAbvGr: Number of bedrooms above basement level.

FullBath / HalfBath: Number of bathrooms.

OverallQual: Overall material and finish quality (added to improve accuracy).

🚀 Implementation Steps
Data Loading: Reading train.csv and test.csv.

Feature Engineering: Combined FullBath and HalfBath into a single TotalBath feature.

Log Transformation: Applied np.log1p to the target variable SalePrice to handle skewness and align with competition evaluation metrics.

Handling Missing Values: Used median imputation to ensure no "NaN" values were passed to the model.

Model Training: Used Scikit-Learn's LinearRegression to fit the model on training data.

Validation: Split the data (80/20) to check performance before making final predictions.

📈 Results
The model achieved a Validation RMSE of: [Insert your RMSE result here, e.g., 0.1824].

📁 Repository Structure
Plaintext

├── data/
│   ├── train.csv
│   └── test.csv
├── task1.py              # Main Python script
├── submission.csv        # Final predictions
└── README.md             # Project documentation



📬 Contact
If you have any questions or want to discuss this project, feel free to reach out!
MANISH KUMAR
