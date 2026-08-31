# tune_and_compare.py
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Dataset
df = pd.read_csv("cleaned_delhi_housing.csv")

if "Price_sqft" in df.columns:
    df = df.drop(columns=["Price_sqft"])

target_col = [c for c in df.columns if 'price' in c.lower()][0]

X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("==============================================")
print("     COMPARING & TUNING ML ALGORITHMS")
print("==============================================\n")

# 2. Quick Algorithm Comparison
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest (Default)": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
}

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    results.append({"Algorithm": name, "R2 Score": r2, "MAE (Lakhs)": mae, "RMSE (Lakhs)": rmse})

# 3. Fast Hyperparameter Tuning
print("Performing Fast Hyperparameter Tuning...")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_dist = {
    'regressor__n_estimators': [50, 100],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5]
}

# n_iter=4 keeps execution extremely fast
random_search = RandomizedSearchCV(
    rf_pipeline, 
    param_distributions=param_dist, 
    n_iter=4, 
    cv=2, 
    scoring='r2', 
    random_state=42, 
    n_jobs=-1
)
random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
tuned_preds = best_rf.predict(X_test)

results.append({
    "Algorithm": "Random Forest (Tuned)",
    "R2 Score": r2_score(y_test, tuned_preds),
    "MAE (Lakhs)": mean_absolute_error(y_test, tuned_preds),
    "RMSE (Lakhs)": np.sqrt(mean_squared_error(y_test, tuned_preds))
})

# 4. Results & Selection
comparison_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

print("\n==============================================")
print("             MODEL COMPARISON")
print("==============================================")
print(comparison_df.to_string(index=False))

best_model_name = comparison_df.iloc[0]["Algorithm"]
print(f"\nBest Model: {best_model_name}")

joblib.dump(best_rf, 'delhi_house_model.pkl')
comparison_df.to_csv("model_comparison.csv", index=False)

print("\n✅ Fast tuning complete! Saved model to 'delhi_house_model.pkl'")