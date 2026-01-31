# Loan Approval Prediction API

This project predicts whether a loan will be approved based on applicant financial and personal details.  
A Machine Learning model is trained using preprocessing pipelines and deployed as a REST API using **FastAPI**.  
Predictions are stored in a **SQLite database** and the API is deployed on **Render**.

Live API:  
https://loan-approval-ml-api-1.onrender.com  

Swagger UI:  
https://loan-approval-ml-api-1.onrender.com/docs  

---

## Project Description

- A dataset of loan applicants is used to train a **RandomForestClassifier**.
- Data preprocessing includes handling missing values, scaling numeric features, and encoding categorical features.
- A pipeline combines preprocessing and model training.
- The trained model is saved and loaded in a FastAPI application.
- Users send applicant details as JSON and receive:
  - Loan approval decision (Yes/No)  
  - Probability score  
  - Top contributing factors

---

## API Endpoint

**POST /predict**

Input: Applicant income, credit score, loan amount, employment status, and other financial details.  
Output: Loan approval result with probability.

---

## Technologies Used

Python, Pandas, Scikit-learn, FastAPI, SQLite, Uvicorn, Render.

---

## Author

Sumedh Bodke
