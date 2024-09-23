import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('assignments_1/Electricity BILL.csv')

# List of numerical features
numerical_features = [
    'Construction_Year', 'Number_of_Floors', 'Energy_Consumption_Per_SqM',
    'Indoor_Air_Quality', 'Smart_Devices_Count', 'Energy_Per_SqM',
    'Number_of_Residents', 'Electricity_Bill', 'Maintenance_Resolution_Time',
    'Green_Certified', 'Water_Usage_Per_Building', 'Waste_Recycled_Percentage',
    'Occupancy_Rate'
]

# Split the numerical features into two halves
# half_size = len(numerical_features) // 2
# first_half_features = numerical_features[:half_size]
# second_half_features = numerical_features[half_size:]

# # Pair Plot for the first half of features
# sns.pairplot(df[first_half_features])
# plt.suptitle('Pair Plot - First Half of Features', y=1.02)
# plt.show()

# # Pair Plot for the second half of features
# sns.pairplot(df[second_half_features])
# plt.suptitle('Pair Plot - Second Half of Features', y=1.02)
# plt.show()

# Pair Plot
sns.pairplot(df,diag_kind="hist")
plt.show()
# Box Plots for Numerical Features

# Plotting the first 6 box plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows and 3 columns for 6 plots

for i, feature in enumerate(numerical_features[:6]):  # First 6 features
    ax = axes[i // 3, i % 3]  # Access axes in 2x3 grid
    sns.boxplot(x='Building_Type', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot of {feature} by Building Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Plotting the remaining 7 box plots
fig, axes = plt.subplots(3, 3, figsize=(18, 18))  # 3 rows and 3 columns for 7 plots

for i, feature in enumerate(numerical_features[6:]):  # Remaining features
    ax = axes[i // 3, i % 3]  # Access axes in 3x3 grid
    sns.boxplot(x='Building_Type', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot of {feature} by Building Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Remove any empty subplots
for j in range(i + 1, 9):
    fig.delaxes(axes[j // 3, j % 3])

plt.tight_layout()
plt.show()

# Create a grid of 6 violin plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows and 3 columns for 6 plots

for i, feature in enumerate(numerical_features[:6]):  # First 6 features
    ax = axes[i // 3, i % 3]  # Access the appropriate subplot
    sns.violinplot(x='Building_Type', y=feature, data=df, ax=ax)
    ax.set_title(f'Violin Plot of {feature} by Building Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Create a grid of 7 violin plots
fig, axes = plt.subplots(3, 3, figsize=(18, 18))  # 3 rows and 3 columns for 7 plots

for i, feature in enumerate(numerical_features[6:]):  # Remaining features
    ax = axes[i // 3, i % 3]  # Access the appropriate subplot
    sns.violinplot(x='Building_Type', y=feature, data=df, ax=ax)
    ax.set_title(f'Violin Plot of {feature} by Building Type')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Remove the unused subplot (in this case, one subplot will remain empty)
for j in range(i + 1, 9):
    fig.delaxes(axes[j // 3, j % 3])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Count Plots for Categorical Features with One-Hot Encoding
categorical_features = ['Building_Type', 'Green_Certified', 'Building_Status', 'Maintenance_Priority']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

for feature in df_encoded.columns:
    if 'Building_Type' in feature or 'Green_Certified' in feature or 'Building_Status' in feature or 'Maintenance_Priority' in feature:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df_encoded)
        plt.title(f'Count Plot of {feature}')
        plt.xticks(rotation=45)
        plt.show()

# Correlation Heatmap for Numerical Features
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Splitting the dataset into training and testing sets (80:20)
X = df[numerical_features]
y = df['Electricity_Bill']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
