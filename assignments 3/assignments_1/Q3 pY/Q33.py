import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("assignments_1/Electricity BILL.csv")


numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

# Step 1: Normalize numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=["Electricity_Bill"])  # Assuming "Electricity_Bill" is the target
y = data["Electricity_Bill"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

def adjusted_r2(r2, X, y):
    """Calculate Adjusted R²."""
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_adj_r2 = adjusted_r2(train_r2, X_train, y_train)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_adj_r2 = adjusted_r2(test_r2, X_test, y_test)

print("\nTrain Metrics:")
print(f"MSE: {train_mse:.6f}")
print(f"RMSE: {train_rmse:.6f}")
print(f"MAE: {train_mae:.6f}")
print(f"R²: {train_r2:.6f}")
print(f"Adjusted R²: {train_adj_r2:.6f}")

print("\nTest Metrics:")
print(f"MSE: {test_mse:.6f}")
print(f"RMSE: {test_rmse:.6f}")
print(f"MAE: {test_mae:.6f}")
print(f"R²: {test_r2:.6f}")
print(f"Adjusted R²: {test_adj_r2:.6f}")
