import pandas as pd

# Load the raw data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID since it’s not useful for prediction
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Handle missing values
df = df.fillna(0)

# Convert TotalCharges to numeric safely
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Save the cleaned dataset
df_encoded.to_csv("cleaned_data.csv", index=False)

print("✅ Cleaned dataset saved as cleaned_data.csv")
