import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load your dataset
data = pd.read_csv('/Users/vishaljha/Desktop/Machine-Learning-IIITD/assignments/Electricity BILL.csv')

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])  # assign directly to avoid the warning
    else:
        data[col] = data[col].fillna(data[col].mean())  # assign directly to avoid the warning

# Encode categorical features
label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

# Separate features and target
X = data.drop('Electricity_Bill', axis=1)
y = data['Electricity_Bill']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Apply Recursive Feature Elimination (RFE)
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X_train, y_train)

# Get the selected feature indices and their names
selected_features = X.columns[rfe.support_]
print("Top 3 most important features:", selected_features)

# Train the regression model on selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe, y_train)

# Predict on the train and test sets
y_train_pred = model.predict(X_train_rfe)
y_test_pred = model.predict(X_test_rfe)

# Calculate performance metrics
def print_metrics(y_true, y_pred, dataset_name=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - X_train_rfe.shape[1] - 1)
    
    print(f"{dataset_name} MSE: {mse}")
    print(f"{dataset_name} RMSE: {rmse}")
    print(f"{dataset_name} MAE: {mae}")
    print(f"{dataset_name} R²: {r2}")
    print(f"{dataset_name} Adjusted R²: {adjusted_r2}")

# Print results for train set
print_metrics(y_train, y_train_pred, "Train")

# Print results for test set
print_metrics(y_test, y_test_pred, "Test")
