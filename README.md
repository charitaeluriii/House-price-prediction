# House-price-prediction
This is my first ever project , I've been exploring machine learning for quite a few days now , I've learnt about linear regression and i wanted to implement it on my own over a very common example like the house price prediction , i will be using the concept of linear regression to create machine learning model to predict house prices. I will be training the model by using the california housing data set from scikit learn and accordingly predicts the housing prices of desired qualities. 



# Libraries to be used
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
# Load California housing data
housing_data = fetch_california_housing()
# Convert to a DataFrame for easier manipulation
df = pd.DataFrame(data=housing_data.data, columns=housing_data.feature_names)
df['Target'] = housing_data.target
print(df.head())
print(df.describe())
X = df.drop(columns='Target')  # Features
y = df['Target']               # Target variable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
