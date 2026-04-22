import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- SET UP PAGE CONFIG FIRST ---
st.set_page_config(page_title="AI Loan Predictor", page_icon="🏦", layout="wide")

# --- ROBUST MODEL LOADING ---
@st.cache_resource
def load_model():
    # This finds the directory where app.py is located
    base_path = os.path.dirname(__file__)
    # Joins the directory with the filename
    model_path = os.path.join(base_path, 'loan_model.pkl')
    
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_model()

# --- UI HEADER ---
st.title("🏦 AI-Powered Loan Predictor")

if model is None:
    st.error("⚠️ 'loan_model.pkl' not found in the repository!")
    st.info("Make sure you have uploaded the 'loan_model.pkl' file to the same folder as this script on GitHub.")
    st.stop()

st.write("Enter your financial details below to estimate your approval odds.")
st.markdown("---")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Details")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, value=2500000, step=100000)
    income = st.number_input("Your Annual Income (₹)", min_value=50000, value=800000, step=50000)
    co_applicant_income = st.number_input("Co-Applicant Annual Income (₹)", min_value=0, value=0, step=50000)
    debt = st.number_input("Current Monthly Debt (₹)", min_value=0, value=15000, step=1000)
    property_value = st.number_input("Total Property Value (₹)", min_value=0, value=0, step=100000)

with col2:
    st.subheader("Personal Details")
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    employment_str = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("Calculate Approval Odds", type="primary", use_container_width=True):
    
    emp_map = {"Unemployed": 0, "Salaried": 1, "Self-Employed": 2}
    
    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'loan_amount': [loan_amount],
        'income': [income],
        'co_applicant_income': [co_applicant_income],
        'debt': [debt],
        'property_value': [property_value],
        'credit_score': [credit_score],
        'employment_status': [emp_map[employment_str]]
    })
    
    try:
        # Get probability
        prediction_array = model.predict_proba(input_data)
        approval_prob = int(round(prediction_array[0][1] * 100))
        
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
                
        with res_col2:
            st.subheader("Household Financial Health")
            total_monthly_income = (income + co_applicant_income) / 12
            remaining_income = max(0, total_monthly_income - debt)
            
            chart_data = pd.DataFrame({
                "Category": ["Monthly Debt", "Remaining Income"],
                "Amount (₹)": [debt, remaining_income]
            })
            
            fig = px.pie(chart_data, values="Amount (₹)", names="Category", 
                         color="Category", 
                         color_discrete_map={"Monthly Debt":"#EF553B", "Remaining Income":"#00CC96"},
                         hole=0.4) 
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
