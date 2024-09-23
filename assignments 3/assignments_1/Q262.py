import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define sigmoid and cost functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta, reg_type='none', reg_param=0.0):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # To avoid log(0)

    # Cost function
    cost = (-1/m) * (y.dot(np.log(h + epsilon)) + (1 - y).dot(np.log(1 - h + epsilon)))

    # L1 Regularization (Lasso)
    if reg_type == 'l1':
        cost += reg_param * np.sum(np.abs(theta[1:])) / m
    # L2 Regularization (Ridge)
    elif reg_type == 'l2':
        cost += reg_param * np.sum(np.square(theta[1:])) / (2*m)

    return cost

# Accuracy function
def compute_accuracy(X, y, theta):
    predictions = sigmoid(X.dot(theta)) >= 0.5
    return np.mean(predictions == y)

# Early stopping criteria
def early_stopping(valid_loss_history, patience):
    # Stop if the validation loss hasn't improved in 'patience' iterations
    if len(valid_loss_history) > patience:
        recent_losses = valid_loss_history[-patience:]
        if min(recent_losses) == valid_loss_history[-patience]:
            return True
    return False

# Mini-Batch Gradient Descent with early stopping and regularization
def mini_batch_gradient_descent(X_train, y_train, X_val, y_val, theta, alpha, iterations, batch_size, patience, reg_type='none', reg_param=0.0):
    m = len(y_train)
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for j in range(0, m, batch_size):
            end = min(j + batch_size, m)
            X_batch = X_train_shuffled[j:end]
            y_batch = y_train_shuffled[j:end]
            gradient = (1 / len(y_batch)) * X_batch.T.dot(sigmoid(X_batch.dot(theta)) - y_batch)

            if reg_type == 'l1':
                gradient[1:] += reg_param * np.sign(theta[1:]) / len(y_batch)  # L1
            elif reg_type == 'l2':
                gradient[1:] += reg_param * theta[1:] / len(y_batch)  # L2

            theta -= alpha * gradient

        # Compute costs and accuracy for tracking
        train_loss_history.append(compute_cost(X_train, y_train, theta, reg_type, reg_param))
        valid_loss_history.append(compute_cost(X_val, y_val, theta, reg_type, reg_param))
        train_acc_history.append(compute_accuracy(X_train, y_train, theta))
        valid_acc_history.append(compute_accuracy(X_val, y_val, theta))

        # Check for early stopping
        if early_stopping(valid_loss_history, patience):
            print(f"Early stopping at iteration {i}")
            break

        if i % 100 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss_history[-1]}, Valid Loss = {valid_loss_history[-1]}, Train Acc = {train_acc_history[-1]}, Valid Acc = {valid_acc_history[-1]}")

    return theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history

# Load your dataset (assuming you've already cleaned and normalized your data)
# Replace 'NA' with NaN and handle missing values
data = np.genfromtxt('/Users/vishaljha/Desktop/Machine-Learning-IIITD/assignments/assignments_1/HeartDisease.csv', delimiter=',', skip_header=1, missing_values='NA', filling_values=np.nan)

# Optionally, handle missing values (e.g., drop rows or columns with NaN)
# Here, we will drop rows with any NaN values for simplicity.
data = data[~np.isnan(data).any(axis=1)]

# Features (X) and target (y)
X = data[:, :-1]  # Exclude 'HeartDisease' from features
y = data[:, -1]   # Target variable 'HeartDisease'

# Add intercept term (bias)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize parameters
theta = np.zeros(X_train.shape[1])
alpha = 0.001  # Learning rate
iterations = 2000
batch_size = 32
patience = 100  # Early stopping patience

# Test different regularization techniques and learning rates
regularization_types = ['none', 'l1', 'l2']
reg_params = [0, 0.01, 0.1, 1]
learning_rates = [0.001, 0.01, 0.1]

# Store results for plotting
results = {}

for reg_type in regularization_types:
    for reg_param in reg_params:
        for lr in learning_rates:
            print(f"Running Mini-Batch Gradient Descent with {reg_type} regularization, reg_param={reg_param}, learning_rate={lr}")
            theta = np.zeros(X_train.shape[1])
            theta, train_loss, valid_loss, train_acc, valid_acc = mini_batch_gradient_descent(
                X_train, y_train, X_val, y_val, theta, lr, iterations, batch_size, patience, reg_type, reg_param
            )
            results[f"{reg_type}_reg={reg_param}_lr={lr}"] = (train_loss, valid_loss, train_acc, valid_acc)


# Plot results
plt.figure(figsize=(18, 12))

# Plot Validation Loss vs. Iteration for each setup
for key, (train_loss, valid_loss, train_acc, valid_acc) in results.items():
    plt.plot(valid_loss, label=key)

plt.title('Validation Loss with Early Stopping (Different Learning Rates and Regularization)')
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
