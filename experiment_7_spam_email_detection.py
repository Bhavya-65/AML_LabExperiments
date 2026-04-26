# Experiment 7: Spam Email Detection

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset
df = pd.read_csv("spam_ham_dataset.csv")   # your file name

print("Dataset Loaded Successfully")
print(df.head())
print("\nColumns:")
print(df.columns)

# Step 2: Use correct columns
X = df["text"]          # messages
y = df["label_num"]     # already encoded (0/1)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Convert text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test_vec)

# Step 7: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Spam Email Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Custom Test
message = ["You have won a free lottery prize!"]
message_vec = vectorizer.transform(message)
prediction = model.predict(message_vec)

print("\nCustom Message Prediction:")
print("Spam" if prediction[0] == 1 else "Not Spam")