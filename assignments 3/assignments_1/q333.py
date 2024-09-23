data = pd.read_csv("Electricity BILL.csv")

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])  
    else:
        data[col] = data[col].fillna(data[col].mean())  

label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

X = data.drop('Electricity_Bill', axis=1)
y = data['Electricity_Bill']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


def print_metrics(y_true, y_pred, dataset_name=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    print(f"{dataset_name} MSE: {mse}")
    print(f"{dataset_name} RMSE: {rmse}")
    print(f"{dataset_name} MAE: {mae}")
    print(f"{dataset_name} R²: {r2}")
    print(f"{dataset_name} Adjusted R²: {adjusted_r2}")
    
    
print_metrics(y_train, y_train_pred, "Train")
print_metrics(y_test, y_test_pred, "Test")
