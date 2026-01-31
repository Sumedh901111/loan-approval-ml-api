# Loan Approval ML API

An end-to-end Machine Learning project that predicts whether a loan should be approved based on applicant details.  
The model is served using FastAPI and provides probability-based decisions with explainability.

---

## 🚀 Features
- Trained ML classification model (Random Forest)
- REST API built with FastAPI
- Probability-based threshold decision
- Top feature importance returned for explainability
- Ready for deployment

---

## 🧠 Tech Stack
- Python
- Scikit-learn
- Pandas
- FastAPI
- Uvicorn

---

## 📦 Project Files
- `app.py` → FastAPI application
- `loan_model.pkl` → trained ML model
- `feature_importance.csv` → top important features
- `requirements.txt` → dependencies

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
