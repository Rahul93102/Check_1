import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("assignments_1/Electricity BILL.csv")


# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

# Step 2: Normalize numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])


# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 2: Split data into train and test sets
X = data.drop(columns=["Electricity_Bill"])  # Assuming "Electricity_Bill" is the target
y = data["Electricity_Bill"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# Predictions
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

def adjusted_r2(r2, X):
    return 1 - (1 - r2) * (len(y_train) - 1) / (len(y_train) - X.shape[1] - 1)

train_adj_r2 = adjusted_r2(train_r2, X_train)
test_adj_r2 = adjusted_r2(test_r2, X_test)

# Report metrics
print(f"Gradient Boosting Regressor Results:")
print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
print(f"Train R²: {train_r2}, Test R²: {test_r2}")
print(f"Train Adjusted R²: {train_adj_r2}, Test Adjusted R²: {test_adj_r2}")
