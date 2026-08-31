# predict.py
import pandas as pd
import joblib

# 1. Load Model Pipeline & Cleaned Dataset
model = joblib.load("delhi_house_model.pkl")
df = pd.read_csv("cleaned_delhi_housing.csv")

if "Price_sqft" in df.columns:
    df = df.drop(columns=["Price_sqft"])

target_col = [c for c in df.columns if 'price' in c.lower()][0]
feature_df = df.drop(columns=[target_col])

# 2. Build Sample Input Safely
sample_property = {}

for col in feature_df.columns:
    # Try converting column to numeric
    converted_col = pd.to_numeric(feature_df[col], errors='coerce')
    
    if converted_col.notna().sum() > 0:
        # It's a numerical column
        sample_property[col] = float(converted_col.median())
    else:
        # It's a categorical column
        sample_property[col] = str(feature_df[col].mode()[0])

# Convert input to DataFrame for model prediction
input_df = pd.DataFrame([sample_property])

# 3. Generate Prediction
predicted_price = model.predict(input_df)[0]

print("\n==============================================")
print("           SINGLE PROPERTY PREDICTION")
print("==============================================")
print("\nInput Features:")
for key, value in sample_property.items():
    print(f"  • {key}: {value}")

print(f"\nEstimated Market Value: ₹ {predicted_price:.2f} Lakhs")
print("==============================================\n")