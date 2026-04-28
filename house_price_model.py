import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load Data
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nFirst 5 rows:")
print(train_df.head())

#Select Features
FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
TARGET   = "SalePrice"

print("\nMissing values in selected features:")
print(train_df[FEATURES + [TARGET]].isnull().sum())

# ── 3. Prepare Train Data
X = train_df[FEATURES].fillna(train_df[FEATURES].median())
y = train_df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scale Features
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

#Train Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#Evaluate
y_pred = model.predict(X_val_scaled)

mae  = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2   = r2_score(y_val, y_pred)

print("\n── Validation Metrics ──────────────────")
print(f"  MAE  : ${mae:,.2f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R²   : {r2:.4f}")
print("────────────────────────────────────────")

# Coefficients 
coef_df = pd.DataFrame({
    "Feature"    : FEATURES,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\nModel Coefficients:")
print(coef_df.to_string(index=False))
print(f"Intercept: {model.intercept_:,.2f}")

#Plots 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("House Price - Linear Regression", fontsize=14, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_val, y_pred, alpha=0.5, color="steelblue", edgecolors="white", linewidths=0.3)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r-", lw=2)
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Actual vs Predicted")

# Residuals
residuals = y_val - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, color="coral", edgecolors="white", linewidths=0.3)
axes[1].axhline(0, color="black", lw=1.5, linestyle="-")
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")
axes[1].set_title("Residual Plot")

# Feature Coefficients
colors = ["green" if c > 0 else "red" for c in coef_df["Coefficient"]]
axes[2].barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
axes[2].axvline(0, color="black", lw=1)
axes[2].set_xlabel("Coefficient Value")
axes[2].set_title("Feature Importance (Coefficients)")

plt.tight_layout()
plt.savefig("house_price_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as 'house_price_results.png'")

#Generate Submission
X_test = test_df[FEATURES].fillna(train_df[FEATURES].median())
X_test_scaled = scaler.transform(X_test)
test_preds = model.predict(X_test_scaled)

submission = pd.DataFrame({
    "Id"       : test_df["Id"],
    "SalePrice": test_preds
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved as 'submission.csv'")
print(submission.head())
