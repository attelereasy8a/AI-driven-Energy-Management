# AI-driven-Energy-Management
開発AIを利用したエネルギー管理システムは、スマートホームのエネルギー消費を最適化し、ユーティリティコストを削減します。
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example dataset: 'date_time', 'energy_consumption'
data = pd.read_csv('energy_consumption_data.csv')

# Preprocessing
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour

# Feature and target variables
X = data[['hour']]  # Simplified feature for demo purposes
y = data['energy_consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print(f'Model RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')

# Optimization for appliance usage
def find_optimal_usage_hours(model, start_hour, end_hour):
    hours = np.array(range(start_hour, end_hour+1)).reshape(-1, 1)
    predicted_consumption = model.predict(hours)
    # Find the hour with the lowest predicted consumption
    optimal_hour = hours[np.argmin(predicted_consumption)].item()
    return optimal_hour

optimal_hour = find_optimal_usage_hours(model, 0, 23)
print(f'Optimal hour for using energy-intensive appliances: {optimal_hour}')

# This is a very basic demonstration. A real system would involve more features,
# complex models (e.g., neural networks for time series forecasting), and real-time
# data processing to continuously optimize energy consumption.
