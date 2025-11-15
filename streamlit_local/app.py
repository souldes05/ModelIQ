import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ModelIQ Predictor", layout="centered")

st.title("💡 ModelIQ: Customer Churn Prediction")
st.markdown("Enter customer details below to get a prediction from your FastAPI backend.")

# Input form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "gender_binary": 1 if gender == "Male" else 0,
        "SeniorCitizen": senior_citizen,
        "Partner_binary": 1 if partner == "Yes" else 0,
        "Dependents_binary": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService_binary": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling_binary": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        # Call your FastAPI endpoint
        api_url = "http://localhost:8000/predict"
        response = requests.post(api_url, json=input_data)
        result = response.json()

        if response.status_code == 200 and result.get("success", False):
            prediction = result.get("prediction")
            st.success(f"✅ Prediction: {prediction}")
        else:
            st.error(f"❌ API Error: {result.get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"⚠️ Could not connect to API: {e}")
