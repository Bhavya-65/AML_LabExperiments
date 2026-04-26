# Experiment 1: Data Preparation and Preprocessing

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load Dataset
df = pd.read_csv("IRIS.csv")
print("Dataset is loaded successfully\n")

# Step 3: Display Dataset
print("First 5 rows")
print(df.head())

# Step 4: Dataset Information
print("\nDataset Info:\n")
print(df.info())

# Step 5: Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Step 6: Handle Missing Values

# Fill numerical columns with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical columns with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values handled")

# Step 7: Encode Categorical Data
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nCategorical data encoded")

# Step 8: Feature Scaling
scaler = StandardScaler()

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nData normalized")

# Step 9: Final Output
print("\nProcessed Dataset:\n")
print(df.head())

# Step 10: Save Processed Data
df.to_csv("processed_data.csv", index=False)

print("\nProcessed data saved as processed_data.csv")

