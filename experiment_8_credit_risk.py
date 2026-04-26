# Experiment 8: Credit Risk Assessment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
df = pd.read_csv("credit_risk_dataset.csv")   
print("Dataset Loaded Successfully")
print(df.head())
print("\nColumns:")
print(df.columns)

# Step 2: Handle missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 3: Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = le.fit_transform(df[col])

# Step 4: Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Credit Risk Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 11: Predict first customer
sample = X.iloc[[0]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nCredit Risk Prediction for first record:")
print("High Risk / Default" if prediction[0] == 1 else "Low Risk / No Default")