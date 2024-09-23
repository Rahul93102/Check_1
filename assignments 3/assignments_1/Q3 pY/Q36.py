import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("assignments_1/Electricity BILL.csv")

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()

# Normalize numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)  # Use sparse_output=False for dense matrix output
encoded_cats = encoder.fit_transform(data[categorical_cols])

# Create a DataFrame of the encoded categorical features
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate numerical and encoded categorical features
data_encoded = pd.concat([data[numerical_cols], encoded_df], axis=1)

# Define features (X) and target (y)
X = data_encoded  # Using the concatenated features
y = data["Electricity_Bill"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Apply ElasticNet with different values of alpha
alphas = [0.1, 0.5, 1.0, 1005.0]  # Different values of the mixing parameter
results = {}

def adjusted_r2(r2, X, y):
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

for alpha in alphas:
    print(f"\nElasticNet with alpha = {alpha}")
    
    # Initialize ElasticNet with the current alpha
    elastic_net = ElasticNet(alpha=alpha, random_state=42)
    elastic_net.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = elastic_net.predict(X_train)
    y_test_pred = elastic_net.predict(X_test)
    
    # Metrics calculation
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_adj_r2 = adjusted_r2(train_r2, X_train, y_train)
    test_adj_r2 = adjusted_r2(test_r2, X_test, y_test)
    
    # Store the results for comparison
    results[alpha] = {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_adj_r2': train_adj_r2, 'test_adj_r2': test_adj_r2
    }
    
    # Report metrics with more digits
    print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
    print(f"Train MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
    print(f"Train R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
    print(f"Train Adjusted R²: {train_adj_r2:.6f}, Test Adjusted R²: {test_adj_r2:.6f}")

# Comparison of results
print("\nComparison of results for different alpha values:")
for alpha, metrics in results.items():
    print(f"\nAlpha = {alpha}:")
    print(f"Test MSE: {metrics['test_mse']:.6f}, Test RMSE: {metrics['test_rmse']:.6f}, Test MAE: {metrics['test_mae']:.6f}")
    print(f"Test R²: {metrics['test_r2']:.6f}, Test Adjusted R²: {metrics['test_adj_r2']:.6f}")
