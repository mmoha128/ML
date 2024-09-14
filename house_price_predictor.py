import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
housing = fetch_california_housing()

# Create a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target (house prices) to the DataFrame
df['PRICE'] = housing.target

# Display the first few rows of the dataset
print(df.head())

# Step 3: Split the dataset into features (X) and target (y)
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients and intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

import pickle

# Assuming `model` is your trained model
with open('house_model.pkl', 'wb') as file:
    pickle.dump(model, file)





