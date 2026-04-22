import streamlit as st
import pandas as pd
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib", "scikit-learn", "plotly", "pandas", "numpy"])
import joblib
import os
import plotly.express as px

# --- LOAD THE AI MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists('loan_model.pkl'):
        return joblib.load('loan_model.pkl')
    return None

model = load_model()

# --- WEBSITE SETUP ---
st.set_page_config(page_title="AI Loan Predictor", page_icon="🏦", layout="wide")
st.title("🏦 AI-Powered Loan Predictor")

if model is None:
    st.error("⚠️ Model not found! Please run 'python train_model.py' first.")
    st.stop()

st.write("Enter your financial details below to estimate your approval odds.")
st.markdown("---")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Details")
    # Updated to Rupees with realistic defaults and step sizes
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, value=2500000, step=100000)
    income = st.number_input("Your Annual Income (₹)", min_value=50000, value=800000, step=50000)
    co_applicant_income = st.number_input("Co-Applicant Annual Income (₹)", min_value=0, value=0, step=50000, help="Income of anyone else applying with you.")
    debt = st.number_input("Current Monthly Debt (₹)", min_value=0, value=15000, step=1000)
    property_value = st.number_input("Total Property Value (₹)", min_value=0, value=0, step=100000, help="Value of homes, land, or vehicles you own (Collateral).")

with col2:
    st.subheader("Personal Details")
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    employment_str = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])

st.markdown("---")

# --- PREDICTION AND VISUALS ---
if st.button("Calculate Approval Odds", type="primary", use_container_width=True):
    
    # Format inputs exactly as the AI expects them
    emp_map = {"Unemployed": 0, "Salaried": 1, "Self-Employed": 2}
    input_data = pd.DataFrame({
        'loan_amount': [loan_amount],
        'income': [income],
        'co_applicant_income': [co_applicant_income],
        'debt': [debt],
        'property_value': [property_value],
        'credit_score': [credit_score],
        'employment_status': [emp_map[employment_str]]
    })
    
    # Ask AI for prediction
    try:
        prediction_array = model.predict_proba(input_data)
        approval_prob = int(round(prediction_array[0][1] * 100))
    except ValueError:
        st.error("🚨 Error: The AI model is out of date! Please run 'python train_model.py' in your terminal to update it.")
        st.stop()
    
    # --- DISPLAY RESULTS ---
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("AI Prediction")
        st.metric(label="Probability of Approval", value=f"{approval_prob}%")
        st.progress(approval_prob / 100.0)
        
        if approval_prob >= 70:
            st.success("Great! High chance of approval.")
        elif approval_prob >= 40:
            st.warning("Fair odds. Borderline approval.")
        else:
            st.error("Low odds. High risk application.")
            
        if property_value > loan_amount:
            st.info("💡 Your property value exceeds the loan amount, which acts as excellent collateral and significantly boosts your odds!")
            
    with res_col2:
        st.subheader("Household Financial Health")
        
        # Calculate combined monthly math for the chart
        total_annual_income = income + co_applicant_income
        monthly_income = total_annual_income / 12
        remaining_income = max(0, monthly_income - debt)
        
        # Create an interactive Pie Chart (Updated labels to ₹)
        chart_data = pd.DataFrame({
            "Category": ["Monthly Debt", "Remaining Household Income"],
            "Amount (₹)": [debt, remaining_income]
        })
        
        fig = px.pie(chart_data, values="Amount (₹)", names="Category", 
                     color="Category", color_discrete_map={"Monthly Debt":"#EF553B", "Remaining Household Income":"#00CC96"},
                     hole=0.4) 
        
        st.plotly_chart(fig, use_container_width=True)