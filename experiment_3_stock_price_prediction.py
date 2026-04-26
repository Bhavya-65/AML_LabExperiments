# Experiment 3: Time-Series Model
# Stock Price Prediction using Kaggle Dataset (AAPL.csv)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("Tesla.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

print("\nColumns:")
print(df.columns)

# Step 3: Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 4: Use only Close Price
df = df[['Close']]

# Step 5: Create Prediction Column (Next Day Price)
df['Prediction'] = df['Close'].shift(-1)

# Step 6: Remove last row (NaN)
df = df.dropna()

# Step 7: Features and Target
X = df[['Close']]
y = df['Prediction']

# Step 8: Train-Test Split (NO shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 9: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Predict
y_pred = model.predict(X_test)

# Step 11: Evaluation
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 12: Actual vs Predicted
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(result.head())

# Step 13: Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Step 14: Predict Next Day Price
last_price = df[['Close']].iloc[[-1]]
next_day_price = model.predict(last_price)

print("\nPredicted Next Day Price:")
print(next_day_price[0])