import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # حل مشکل نمایش در برخی سیستم‌ها
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

# بارگذاری مدل غیرخطی (مثلاً RandomForestRegressor) از فایل
model = joblib.load("random_forest_model.pkl")
print("Model loaded successfully.")

# بارگذاری داده‌های تست
df_test = pd.read_csv("output.csv")
X_test = df_test[["x", "y", "width", "height"]].values
y_test = df_test["radar_distance"].values

# انجام پیش‌بینی توسط مدل
y_pred = model.predict(X_test)

# چاپ مقادیر واقعی و پیش‌بینی‌شده در کنار هم
print(f"{'Actual':<15} {'Predicted'}")
for actual, predicted in zip(y_test, y_pred.astype(int)):
    print(f"{actual:<15} {predicted}")
    
# محاسبه MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# نمودار مقدار واقعی در برابر مقدار پیش‌بینی شده
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", edgecolors="k")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # خط ایده‌آل y = x
plt.xlabel("Actual Radar Distance")
plt.ylabel("Predicted Radar Distance")
plt.title("Actual vs. Predicted")
plt.grid()
plt.savefig("actual_vs_predicted.png")  # ذخیره نمودار
plt.show()

# هیستوگرام خطاها
errors = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=20, color="purple", alpha=0.7, edgecolor="black")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.grid()
plt.savefig("histogram_errors.png")  # ذخیره نمودار
plt.show()
