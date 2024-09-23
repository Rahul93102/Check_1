import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('assignments_1/Electricity BILL.csv')

# Define numerical features and target
numerical_features = [
    'Construction_Year', 'Number_of_Floors', 'Energy_Consumption_Per_SqM',
    'Water_Usage_Per_Building', 'Waste_Recycled_Percentage', 'Occupancy_Rate',
    'Indoor_Air_Quality', 'Smart_Devices_Count', 'Energy_Per_SqM', 'Number_of_Residents',
    'Electricity_Bill'
]

# Splitting the dataset
X = df[numerical_features]
y = df['Electricity_Bill']

# Step 1: Standardize the numerical features


# Step 2: Apply UMAP for dimensionality reduction with adjusted parameters
umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1)
X_reduced = umap_reducer.fit_transform(X)

# Step 3: Create a scatter plot of the reduced data
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.colorbar(scatter)
plt.title('UMAP Projection of the Electricity Bill Dataset')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.show()
