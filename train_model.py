import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Generating updated training data (INR)...")

# Updated historical data using realistic Rupee amounts
data = {
    'loan_amount': [5000000, 1000000, 20000000, 500000, 15000000, 3000000, 40000000],
    'income': [1200000, 400000, 2500000, 200000, 3500000, 600000, 1500000],
    'co_applicant_income': [0, 0, 1000000, 0, 1500000, 300000, 2000000],
    'debt': [25000, 10000, 60000, 5000, 40000, 15000, 80000],
    'property_value': [0, 500000, 30000000, 0, 0, 1000000, 60000000],
    'credit_score': [700, 600, 650, 750, 800, 550, 680],
    'employment_status': [1, 0, 1, 1, 2, 0, 1], 
    'approved': [1, 0, 1, 1, 1, 0, 1] 
}

df = pd.DataFrame(data)

# Separate inputs (X) from the target (y)
X = df[['loan_amount', 'income', 'co_applicant_income', 'debt', 'property_value', 'credit_score', 'employment_status']]
y = df['approved']

print("Training the AI model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'loan_model.pkl')
print("✅ Success! The new 'loan_model.pkl' has been saved with INR data.")