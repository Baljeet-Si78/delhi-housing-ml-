import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "Delhi_v2.csv"

df = pd.read_csv(file_path)

print("\n========================================")
print("       DELHI HOUSING DATA CLEANING")
print("========================================")

print("\nOriginal Dataset Shape:", df.shape)

print("\nOriginal Columns:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 3. SELECT RELEVANT COLUMNS
# ============================================================

columns_to_keep = [
    'price',
    'Address',
    'area',
    'latitude',
    'longitude',
    'Bedrooms',
    'Bathrooms',
    'balcony',
    'Status',
    'neworold',
    'parking',
    'Furnished_status',
    'Lift',
    'Landmark',
    'type_of_building',
    'Price_sqft'
]

# Keep only columns that actually exist

available_columns = [
    col for col in columns_to_keep
    if col in df.columns
]

df = df[available_columns].copy()


print("\nColumns Being Used:")
print(df.columns.tolist())


# ============================================================
# 4. CLEAN PRICE
# ============================================================

if 'price' not in df.columns:
    raise ValueError(
        "ERROR: 'price' column was not found!"
    )

# Convert price to numeric

df['price'] = (
    df['price']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['price'] = pd.to_numeric(
    df['price'],
    errors='coerce'
)


# ============================================================
# 5. CLEAN AREA
# ============================================================

if 'area' in df.columns:

    df['area'] = (
        df['area']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )

    df['area'] = pd.to_numeric(
        df['area'],
        errors='coerce'
    )


# ============================================================
# 6. CLEAN NUMERICAL COLUMNS
# ============================================================

numerical_columns = [
    'latitude',
    'longitude',
    'Bedrooms',
    'Bathrooms',
    'balcony',
    'parking',
    'Lift',
    'Price_sqft'
]

for col in numerical_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )


# ============================================================
# 7. REMOVE ROWS WITH MISSING PRICE OR AREA
# ============================================================

if 'area' in df.columns:

    df = df.dropna(
        subset=['price', 'area']
    )

else:

    df = df.dropna(
        subset=['price']
    )


# ============================================================
# 8. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

categorical_columns = [
    'Address',
    'Status',
    'neworold',
    'Furnished_status',
    'Landmark',
    'type_of_building'
]

for col in categorical_columns:

    if col in df.columns:

        df[col] = df[col].fillna(
            'Unknown'
        )


# ============================================================
# 9. HANDLE MISSING NUMERICAL VALUES
# ============================================================

for col in numerical_columns:

    if col in df.columns:

        if df[col].notna().any():

            df[col] = df[col].fillna(
                df[col].median()
            )


# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(
    "\nDuplicate rows removed:",
    before - after
)


# ============================================================
# 11. CREATE PRICE IN LAKHS
# ============================================================

# Keep original price and create a model-friendly
# target column in Lakhs.

df['Price_Lakhs'] = df['price'] / 100000


# ============================================================
# 12. REMOVE ORIGINAL PRICE
# ============================================================

df = df.drop(
    columns=['price']
)


# ============================================================
# 13. FINAL DATASET INFORMATION
# ============================================================

print("\n========================================")
print("        CLEANED DATASET")
print("========================================")

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 14. SAVE CLEANED DATASET
# ============================================================

output_file = "cleaned_delhi_housing.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("Cleaning completed successfully!")
print("Saved as:", output_file)
print("========================================")