import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('assignments_1/Electricity BILL.csv')

# One-Hot Encode categorical features
categorical_features = ['Building_Type', 'Green_Certified', 'Building_Status', 'Maintenance_Priority']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define numerical and target features
numerical_features = [
    'Construction_Year', 'Number_of_Floors', 'Energy_Consumption_Per_SqM',
    'Water_Usage_Per_Building', 'Waste_Recycled_Percentage', 'Occupancy_Rate',
    'Indoor_Air_Quality', 'Smart_Devices_Count', 'Energy_Per_SqM', 'Number_of_Residents',
]
X = df_encoded[numerical_features]
y = df_encoded['Electricity_Bill']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Adjusted R2 Score calculation
    n = len(y_true)
    p = X_train.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return mse, rmse, r2, adjusted_r2, mae

# Perform ICA and evaluate models for different numbers of components
components_list = [4, 5, 6, 8]
results = {}

for n_components in components_list:
    # Perform ICA
    ica = FastICA(n_components=n_components, random_state=42)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    
    # Train a Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train_ica, y_train)
    
    # Predictions
    y_train_pred = lr.predict(X_train_ica)
    y_test_pred = lr.predict(X_test_ica)
    
    # Calculate metrics for training and testing datasets
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Store the results
    results[n_components] = {'train': train_metrics, 'test': test_metrics}

# Display the results
for n_components, metrics in results.items():
    print(f"\nResults for {n_components} ICA Components:")
    print("Train Metrics (MSE, RMSE, R2, Adjusted R2, MAE):", metrics['train'])
    print("Test Metrics (MSE, RMSE, R2, Adjusted R2, MAE):", metrics['test'])
