import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load CSV file
csv_file = "output.csv"  
df = pd.read_csv(csv_file)
# Optional: Filter to only include valid radar_distance values (e.g., between 20 and 90)
# df = df[(df["radar_distance"] >= 20) & (df["radar_distance"] <= 90)]

# Features (input): x, y, width, height
X = df[["x", "y", "width", "height"]]
y = pd.to_numeric(df["radar_distance"], errors="coerce")  # Convert target to numeric

# Split data into training (95%) and testing (5%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=40)

# Optionally, you can scale the features if needed
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Train a non-linear model: RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=40)
model.fit(X_train, y_train)

# Save the model as a pickle file
joblib.dump(model, "random_forest_model.pkl")
print("Model saved to random_forest_model.pkl")

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Instead of coefficients, we look at feature importances
print("Feature Importances:", model.feature_importances_)

# Example: Predict radar_distance for a new input
new_input = [[417.8, 263.0, 40.2, 26.0]]  # Example values for x, y, width, height
predicted_distance = model.predict(new_input)
print(f"Predicted radar_distance: {predicted_distance[0]:.4f}")

# Save feature importances to a text file
with open("model_feature_importances.txt", "w") as f:
    f.write("Feature Importances: " + ", ".join(map(str, model.feature_importances_)) + "\n")
print("Feature importances saved to model_feature_importances.txt")

# Save test data (X_test and y_test) to a CSV file
test_data = X_test.copy()
test_data['radar_distance'] = y_test
test_data.to_csv("test_data.csv", index=False)
print("Test data saved to test_data.csv")
