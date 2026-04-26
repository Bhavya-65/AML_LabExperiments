# Experiment 5: Handwritten Digit Recognition using MNIST

# Step 1: Import Libraries
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns

# Step 2: Load Dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Dataset Loaded Successfully")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 3: Display First Image
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Digit: {digits.target[0]}")
plt.show()

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Prediction
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("MNIST Digit Recognition Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Test One Prediction
index = 0
sample = X_test[index].reshape(1, -1)
prediction = model.predict(sample)

print("\nPredicted Digit:", prediction[0])