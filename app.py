# app.py
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Delhi Real Estate Valuation Engine",
    page_icon="🏠",
    layout="centered"
)

# Load Model Pipeline & Cleaned Dataset
@st.cache_resource
def load_assets():
    model = joblib.load("delhi_house_model.pkl")
    df = pd.read_csv("cleaned_delhi_housing.csv")
    if "Price_sqft" in df.columns:
        df = df.drop(columns=["Price_sqft"])
    return model, df

model, df = load_assets()

# UI Title & Header
st.title("🏙️ Delhi Housing Price Predictor")
st.write("Estimate market property values across Delhi-NCR using Machine Learning.")
st.markdown("---")

st.sidebar.header("📊 Enter Property Features")

# Build Inputs Dynamically
input_data = {}
target_col = [c for c in df.columns if 'price' in c.lower()][0]
feature_df = df.drop(columns=[target_col])

for col in feature_df.columns:
    converted_col = pd.to_numeric(feature_df[col], errors='coerce')
    
    # Numerical Column
    if converted_col.notna().sum() > 0:
        min_v = float(converted_col.min())
        max_v = float(converted_col.max())
        med_v = float(converted_col.median())
        
        # Use integer step for BHK/Bathrooms, floating for Area
        step_val = 1.0 if 'bhk' in col.lower() or 'bath' in col.lower() else 10.0
        input_data[col] = st.sidebar.number_input(
            f"{col}",
            min_value=min_v,
            max_value=max_v,
            value=med_v,
            step=step_val
        )
    # Categorical Column
    else:
        options = sorted(list(feature_df[col].dropna().unique()))
        input_data[col] = st.sidebar.selectbox(f"{col}", options)

# Main Screen Prediction Trigger
if st.button("🔮 Calculate Valuation", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    predicted_val = model.predict(input_df)[0]
    
    st.markdown("---")
    st.subheader("💡 Estimated Property Price")
    st.success(f"### ₹ {predicted_val:.2f} Lakhs")
    
    # Input Summary Expandable Table
    with st.expander("See Input Details"):
        st.json(input_data)