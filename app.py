from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3

# ---------------------------
# Load model and feature importance
# ---------------------------
model = joblib.load("loan_model.pkl")

feature_importance = pd.read_csv("feature_importance.csv")
feature_importance["Feature"] = (
    feature_importance["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

# ---------------------------
# SQLite connection (cloud-safe)
# ---------------------------
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Applicant_Income REAL,
    Coapplicant_Income REAL,
    Employment_Status TEXT,
    Age INTEGER,
    Marital_Status TEXT,
    Dependents INTEGER,
    Credit_Score INTEGER,
    Existing_Loans INTEGER,
    DTI_Ratio REAL,
    Savings REAL,
    Collateral_Value REAL,
    Loan_Amount REAL,
    Loan_Term INTEGER,
    Loan_Purpose TEXT,
    Property_Area TEXT,
    Education_Level TEXT,
    Total_Income REAL,
    prediction TEXT,
    probability REAL
)
""")
conn.commit()

# ---------------------------
# Input schema
# ---------------------------
class LoanInput(BaseModel):
    Applicant_Income: float
    Coapplicant_Income: float
    Employment_Status: str
    Age: int
    Marital_Status: str
    Dependents: int
    Credit_Score: int
    Existing_Loans: int
    DTI_Ratio: float
    Savings: float
    Collateral_Value: float
    Loan_Amount: float
    Loan_Term: int
    Loan_Purpose: str
    Property_Area: str
    Education_Level: str
    Total_Income: float

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict(data: LoanInput):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict probability
    proba = model.predict_proba(df)[0][1]
    pred = int(proba >= 0.4)
    result = "Yes" if pred == 1 else "No"

    top_factors = feature_importance["Feature"].head(3).tolist()

    # Save to SQLite
    sql = """
    INSERT INTO predictions (
        Applicant_Income, Coapplicant_Income, Employment_Status, Age,
        Marital_Status, Dependents, Credit_Score, Existing_Loans,
        DTI_Ratio, Savings, Collateral_Value, Loan_Amount, Loan_Term,
        Loan_Purpose, Property_Area, Education_Level, Total_Income,
        prediction, probability
    )
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    values = (
        data.Applicant_Income,
        data.Coapplicant_Income,
        data.Employment_Status,
        data.Age,
        data.Marital_Status,
        data.Dependents,
        data.Credit_Score,
        data.Existing_Loans,
        data.DTI_Ratio,
        data.Savings,
        data.Collateral_Value,
        data.Loan_Amount,
        data.Loan_Term,
        data.Loan_Purpose,
        data.Property_Area,
        data.Education_Level,
        data.Total_Income,
        result,
        float(proba)
    )

    cursor.execute(sql, values)
    conn.commit()

    return {
        "Loan_Approved": result,
        "Probability": round(float(proba), 3),
        "Top_Factors": top_factors
    }

# ---------------------------
# Run locally (not used on Render)
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
