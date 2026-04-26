# Experiment 6: Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Dataset
df = pd.read_csv("IRIS.csv")   

print("Dataset Loaded Successfully")
print(df.head())
print("\nColumns:")
print(df.columns)

# Step 2: Drop Id column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Step 3: Encode target column
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Step 4: Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Iris Flower Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 11: Predict one sample
sample = X.iloc[[0]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nPredicted Flower Class:", le.inverse_transform(prediction)[0])