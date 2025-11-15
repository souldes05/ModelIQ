# ~/modeliq/frontend/app.py
import streamlit as st
import requests
import pandas as pd
# Check backend health
def backend_online():
    try:
        r = requests.get("http://localhost:8000/ready", timeout=3)
        return r.status_code == 200
    except:
        return False

if not backend_online():
    st.error("🚨 Backend API is offline. Please start your FastAPI server first.")
    st.stop()


# Set Streamlit page config
st.set_page_config(page_title="ModelIQ Predictor", layout="centered")

# FastAPI backend URL
API_URL = "http://localhost:8000/predict"

st.title("💡 ModelIQ: Customer Churn Prediction")
st.write("Enter customer details below and get a prediction instantly.")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

# --- Prepare data ---
input_data = {
    "gender_binary": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner_binary": 1 if partner == "Yes" else 0,
    "Dependents_binary": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "PhoneService_binary": 1 if phone_service == "Yes" else 0,
    "PaperlessBilling_binary": 1 if paperless == "Yes" else 0,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# --- Predict Button ---
if st.button("Predict Churn"):
    try:
        with st.spinner("Contacting ModelIQ backend..."):
            response = requests.post(API_URL, json={"data": [input_data]})
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("predictions", [])[0]
            st.success(f"✅ Prediction: {'Will Churn' if prediction == 1 else 'Will Stay'}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")

st.caption("Powered by ModelIQ Backend API (FastAPI)")
