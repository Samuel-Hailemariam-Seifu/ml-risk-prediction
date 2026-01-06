# app.py
# ===========================
# Streamlit App for Credit Risk Prediction
# ===========================

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../models/xgb_risk_model.pkl")

# Load dataset for unique values in select boxes
df = pd.read_csv("../data/german_credit_data.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

st.title("Credit Risk Prediction System")
st.write("Predicts whether a customer is high-risk or low-risk.")

# User Input
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex", df["Sex"].unique())
job = st.selectbox("Job", df["Job"].unique())
housing = st.selectbox("Housing", df["Housing"].unique())
saving_accounts = st.selectbox("Saving Accounts", df["Saving accounts"].fillna("unknown").unique())
checking_account = st.selectbox("Checking Account", df["Checking account"].fillna("unknown").unique())
credit_amount = st.number_input("Credit Amount", 0, 100000, 1000)
duration = st.number_input("Duration (Months)", 1, 100, 12)
purpose = st.selectbox("Purpose", df["Purpose"].unique())

input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Job": job,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Credit amount": credit_amount,
    "Duration": duration,
    "Purpose": purpose
}])

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    risk_label = "High Risk (Bad)" if pred == 1 else "Low Risk (Good)"
    st.write(f"**Prediction:** {risk_label}")
    st.write(f"**Probability of Risk:** {prob:.2f}")
st.write("Developed by Samuel Hailemariam Seifu")