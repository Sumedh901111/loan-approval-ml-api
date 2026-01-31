Loan Approval Prediction API

This project predicts whether a loan will be approved based on applicant financial and personal details.
A Machine Learning model is trained using preprocessing pipelines and deployed as a REST API using FastAPI.
Predictions are stored in a SQLite database and the API is deployed on Render.

Live API:
https://loan-approval-ml-api.onrender.com

Swagger UI:
https://loan-approval-ml-api.onrender.com/docs

Model:
RandomForestClassifier with preprocessing (imputation, scaling, encoding).

Features:
Applicant_Income, Coapplicant_Income, Employment_Status, Age, Marital_Status, Dependents, Credit_Score, Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount, Loan_Term, Loan_Purpose, Property_Area, Education_Level, Total_Income.

Target:
Loan_Approved (Yes / No)

Endpoint:
POST /predict

Sample request:

{
"Applicant_Income": 5000,
"Coapplicant_Income": 2000,
"Employment_Status": "Salaried",
"Age": 35,
"Marital_Status": "Married",
"Dependents": 1,
"Credit_Score": 700,
"Existing_Loans": 1,
"DTI_Ratio": 0.3,
"Savings": 8000,
"Collateral_Value": 30000,
"Loan_Amount": 15000,
"Loan_Term": 60,
"Loan_Purpose": "Car",
"Property_Area": "Urban",
"Education_Level": "Graduate",
"Total_Income": 7000
}

Sample response:

{
"Loan_Approved": "No",
"Probability": 0.39,
"Top_Factors": ["Credit_Score", "DTI_Ratio", "Applicant_Income"]
}

Tech stack:
Python, Pandas, Scikit-learn, FastAPI, SQLite, Uvicorn, Render.

Author:
Sumedh Bodke
