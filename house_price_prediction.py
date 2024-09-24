# Title of Project: House Price Prediction using Linear Regression

# Objective: To predict the prices of houses based on various features such as size, location, and age.

# Data Source: You can use any publicly available dataset like the one from Kaggle:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import Data
data = pd.read_csv('housing_prices.csv')

# Describe Data
print("Dataset Information:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())

# Data Visualization
# Pairplot to see relationships between features
sns.pairplot(data)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing
# Handle missing values
data.fillna(method='ffill', inplace=True)  # Forward fill for missing values as an example

# Encode categorical variables if needed
# Assuming 'Neighborhood' is categorical
# data = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)

# Define Target Variable (y) and Feature Variables (X)
X = data.drop('SalePrice', axis=1)  # Features (excluding the target variable)
y = data['SalePrice']  # Target variable (House price)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")

# Prediction
# Example new data input (e.g., size, number of bedrooms, etc.)
# Replace with actual feature values as needed
new_data = [[1500, 3, 2, 1]]  # Example feature set: square footage, bedrooms, bathrooms, etc.
predicted_price = model.predict(new_data)
print(f"Predicted Price for the new house: ${predicted_price[0]:.2f}")

# Explanation:
# The model uses Linear Regression to predict house prices. The MSE shows the model's error in price prediction.
# Improvements: You can try more complex models (like decision trees, random forests) and better data preprocessing.
