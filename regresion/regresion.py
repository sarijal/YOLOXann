import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Load CSV file
csv_file = "output.csv"  
df = pd.read_csv(csv_file)
# df = df[(df["radar_distance"] >= 20) & (df["radar_distance"] <= 90)]

# Features (input): x, y, width, height
X = df[["x", "y", "width", "height"]]
y = pd.to_numeric(df["radar_distance"], errors="coerce")  # Convert target to numeric

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=40)

# Scale features (normalization improves accuracy)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Show model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)



# Example: Predict radar_distance for a new input
new_input = [[417.8,263.0,40.2,26.0]]  # Example values for x, y, width, height
predicted_distance = model.predict(X_test)

print(f"Predicted radar_distance: {predicted_distance[0]:.4f}")



# ذخیره ضرایب و عرض از مبداء در فایل متنی
coefficients = model.coef_
intercept = model.intercept_

# ذخیره در یک فایل متنی
with open("model_coefficients.txt", "w") as f:
    f.write("Coefficients: " + ", ".join(map(str, coefficients)) + "\n")
    f.write("Intercept: " + str(intercept) + "\n")

print("Coefficients and intercept saved to model_coefficients.txt")

# ذخیره داده‌های تست (X_test و y_test) در فایل CSV
test_data = X_test.copy()
test_data['radar_distance'] = y_test

# ذخیره در یک فایل CSV
test_data.to_csv("test_data.csv", index=False)
print("Test data saved to test_data.csv")