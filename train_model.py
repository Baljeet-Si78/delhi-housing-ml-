import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv("cleaned_delhi_housing.csv")

print("\n==============================================")
print("       DELHI HOUSE PRICE PREDICTION")
print("==============================================")

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. DEFINE TARGET
# ============================================================

target = "Price_Lakhs"

if target not in df.columns:
    raise ValueError(
        f"\nERROR: '{target}' column not found in dataset!"
    )

print("\nTarget Variable:", target)


# ============================================================
# 3. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ============================================================

# Price_sqft is removed because:
#
# Price_sqft = Price / Area
#
# Since Price is what we are trying to predict,
# Price_sqft would cause target leakage.

columns_to_remove = [
    "Price_sqft"
]

for col in columns_to_remove:
    if col in df.columns:
        df = df.drop(columns=[col])

print("\nPrice_sqft removed to prevent target leakage.")


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target])

y = df[target]


print("\nNumber of input features:", X.shape[1])


# ============================================================
# 5. IDENTIFY NUMERICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=[
        "int64",
        "int32",
        "float64",
        "float32"
    ]
).columns.tolist()


# ============================================================
# 6. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print("\n----------------------------------------------")
print("NUMERICAL FEATURES")
print("----------------------------------------------")

for feature in numerical_features:
    print("-", feature)


print("\n----------------------------------------------")
print("CATEGORICAL FEATURES")
print("----------------------------------------------")

for feature in categorical_features:
    print("-", feature)


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n----------------------------------------------")
print("DATA SPLIT")
print("----------------------------------------------")

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 8. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ============================================================
# 9. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 10. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ]
)


# ============================================================
# 11. RANDOM FOREST REGRESSOR
# ============================================================

random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


# ============================================================
# 12. COMPLETE MACHINE LEARNING PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            random_forest
        )

    ]
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n==============================================")
print("           TRAINING MODEL")
print("==============================================")

print("\nTraining Random Forest...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

print("\nMaking predictions...")

predictions = model_pipeline.predict(
    X_test
)


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("           MODEL PERFORMANCE")
print("==============================================")

print(f"\nR² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f} Lakhs")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f} Lakhs")

print("\n==============================================")


# ============================================================
# 17. EXAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Price (Lakhs)": y_test.values,
    "Predicted Price (Lakhs)": predictions
})

results["Difference (Lakhs)"] = (
    results["Actual Price (Lakhs)"]
    - results["Predicted Price (Lakhs)"]
)

print("\nFirst 10 Predictions:")
print(
    results.head(10).to_string(index=False)
)


# ============================================================
# 18. SAVE TRAINED MODEL
# ============================================================

model_file = "delhi_house_model.pkl"

joblib.dump(
    model_pipeline,
    model_file
)

print("\n==============================================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================================")

print(
    f"\nSaved as: {model_file}"
)

print("\nYou can now use this model for predictions.")