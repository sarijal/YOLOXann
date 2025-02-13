import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # حل مشکل نمایش در برخی سیستم‌ها
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 📌 خواندن ضرایب و عرض از مبدأ از فایل
with open("model_coefficients.txt", "r") as f:
    lines = f.readlines()
    coefficients = np.array(list(map(float, lines[0].strip().split(":")[1].split(","))))
    intercept = float(lines[1].strip().split(":")[1])

print(f"Loaded coefficients: {coefficients}")
print(f"Loaded intercept: {intercept}")

# 📌 بارگذاری داده‌های تست
# df_test = pd.read_csv("test_data.csv")
df_test = pd.read_csv("output.csv")
X_test = df_test[["x", "y", "width", "height"]].values
y_test = df_test["radar_distance"].values

# 📌 انجام پیش‌بینی
y_pred = np.dot(X_test, coefficients) + intercept

# چاپ مقادیر واقعی و پیش‌بینی‌شده در کنار هم
print(f"{'Actual':<15} {'Predicted'}")
for actual, predicted in zip(y_test, y_pred.astype(int)):  # استفاده از zip برای ترکیب واقعی و پیش‌بینی
    print(f"{actual:<15} {predicted}")
    
# 📌 محاسبه MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# ✅ **1. نمودار مقدار واقعی در برابر مقدار پیش‌بینی شده**
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", edgecolors="k")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # خط ایده‌آل y = x
plt.xlabel("Actual Radar Distance")
plt.ylabel("Predicted Radar Distance")
plt.title("Actual vs. Predicted")
plt.grid()
plt.savefig("actual_vs_predicted.png")  # ذخیره نمودار
plt.show()

# ✅ **2. هیستوگرام خطاها**
errors = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=20, color="purple", alpha=0.7, edgecolor="black")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.grid()
plt.savefig("histogram_errors.png")  # ذخیره نمودار
plt.show()
