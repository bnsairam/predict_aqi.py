# Predicting Air Quality Index (AQI) using Python
# Author: Your Name
# Last Updated: Oct 2025

# ==============================
# 1. Import Required Libraries
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 2. Load the Dataset
# ==============================

# Replace with your dataset path
# Example: 'air_quality_data.csv'
data = pd.read_csv('air_quality_data.csv')
print("\n✅ Dataset Loaded Successfully!\n")
print(data.head())

# ==============================
# 3. Data Preprocessing
# ==============================

# Remove missing values if any
data = data.dropna()

# Normalize column names
data.columns = [col.strip().lower() for col in data.columns]

print("\n🔍 Dataset Info:\n")
print(data.info())
print("\n📊 Basic Statistics:\n")
print(data.describe())

# ==============================
# 4. Exploratory Data Analysis (EDA)
# ==============================

# Pairplot to visualize relationships
sns.pairplot(data)
plt.suptitle("Pairplot of AQI and Pollutants", y=1.02)
plt.show()

# Correlation heatmap
corr = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# 5. Feature Selection
# ==============================

# Selecting relevant features (independent variables)
X = data[['co aqi value', 'ozone aqi value', 'no2 aqi value', 'pm2.5 aqi value']]

# Target variable (dependent)
y = data['aqi value']

# ==============================
# 6. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n📚 Training Samples:", X_train.shape[0])
print("🧪 Testing Samples:", X_test.shape[0])

# ==============================
# 7. Model Training - Random Forest
# ==============================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n🤖 Model Training Complete!\n")

# ==============================
# 8. Model Evaluation
# ==============================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📈 Model Evaluation Metrics:")
print("Mean Absolute Error (MAE):", round(mae, 3))
print("Mean Squared Error (MSE):", round(mse, 3))
print("R² Score:", round(r2, 3))

# ==============================
# 9. Plotting Actual vs Predicted AQI
# ==============================

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual AQI', color='blue')
plt.plot(y_pred, label='Predicted AQI', color='red', alpha=0.7)
plt.title('Actual vs Predicted Air Quality Index (AQI)')
plt.xlabel('Samples')
plt.ylabel('AQI')
plt.legend()
plt.show()

# ==============================
# 10. Save Model (Optional)
# ==============================

import joblib
joblib.dump(model, 'aqi_predictor_model.pkl')
print("\n💾 Model saved as 'aqi_predictor_model.pkl'\n")

print("\n✅ AQI Prediction Pipeline Completed Successfully!")
