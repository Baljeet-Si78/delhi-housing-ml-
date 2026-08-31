# model_analysis.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Dataset and Model
df = pd.read_csv("cleaned_delhi_housing.csv")
model = joblib.load("delhi_house_model.pkl")

# 2. Clean Up Columns & Target
if "Price_sqft" in df.columns:
    df = df.drop(columns=["Price_sqft"])

target_col = [c for c in df.columns if 'price' in c.lower()][0]

X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Train-Test Split & Predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
predictions = model.predict(X_test)

# 4. Metrics Evaluation
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n==============================================")
print("              MODEL ANALYSIS")
print("==============================================")
print(f"\nR² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f} Lakhs")
print(f"RMSE     : {rmse:.4f} Lakhs")

# 5. Extract Steps Safely
preprocessor = model.named_steps["preprocessor"]

# Get the last step dynamically regardless of its key name ('model' or 'regressor')
last_step_name = list(model.named_steps.keys())[-1]
regressor = model.named_steps[last_step_name]

feature_names = preprocessor.get_feature_names_out()
importance = regressor.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\n==============================================")
print("          TOP 20 IMPORTANT FEATURES")
print("==============================================")
print(feature_importance.head(20).to_string(index=False))

# 6. Save Plots
top_features = feature_importance.head(15).sort_values(by="Importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)

plt.figure(figsize=(9, 7))
plt.scatter(y_test, predictions, alpha=0.6, color="teal")
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], color="red", linestyle="--")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)

feature_importance.to_csv("feature_importance.csv", index=False)

print("\n==============================================")
print("              ANALYSIS COMPLETE")
print("==============================================")
print("Generated Files: feature_importance.png, actual_vs_predicted.png, feature_importance.csv")