import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('./assignments_1/HeartDisease.csv')

data.fillna(data.mean(), inplace=True)

X = data.drop("HeartDisease", axis=1).values  
y = data["HeartDisease"].values  

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  
X = (X - X_mean) / X_std

# Add bias term to features
X = np.c_[np.ones(X.shape[0]), X]

# Split data into 70% train, 15% validation, and 15% test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def min_max_scaling(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    range_X = X_max - X_min
    range_X[range_X == 0] = 1  
    return (X - X_min) / range_X

def no_scaling(X):
    return X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5 
    cost = (-1/m) * (y.dot(np.log(h + epsilon)) + (1 - y).dot(np.log(1 - h + epsilon)))
    return cost

def compute_accuracy(X, y, theta):
    predictions = sigmoid(X.dot(theta)) >= 0.5
    return np.mean(predictions == y)

def gradient_descent(X_train, y_train, X_valid, y_valid, theta, alpha, iterations):
    m = len(y_train)
    train_loss_history = np.zeros(iterations)
    valid_loss_history = np.zeros(iterations)
    train_acc_history = np.zeros(iterations)
    valid_acc_history = np.zeros(iterations)

    for i in range(iterations):
        gradient = (1/m) * X_train.T.dot(sigmoid(X_train.dot(theta)) - y_train)
        theta -= alpha * gradient

        train_loss_history[i] = compute_cost(X_train, y_train, theta)
        valid_loss_history[i] = compute_cost(X_valid, y_valid, theta)
        train_acc_history[i] = compute_accuracy(X_train, y_train, theta)
        valid_acc_history[i] = compute_accuracy(X_valid, y_valid, theta)

        if i % 1000 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss_history[i]}, Valid Loss = {valid_loss_history[i]}, Train Acc = {train_acc_history[i]}, Valid Acc = {valid_acc_history[i]}")

    return theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history

alpha = 0.001  
iterations = 30000

X_train_minmax = min_max_scaling(X_train)
X_val_minmax = min_max_scaling(X_val)
X_test_minmax = min_max_scaling(X_test)

theta_minmax = np.zeros(X_train_minmax.shape[1])
theta_minmax, train_loss_history_minmax, valid_loss_history_minmax, train_acc_history_minmax, valid_acc_history_minmax = gradient_descent(
    X_train_minmax, y_train, X_val_minmax, y_val, theta_minmax, alpha, iterations
)

X_train_noscale = no_scaling(X_train)
X_val_noscale = no_scaling(X_val)
X_test_noscale = no_scaling(X_test)

theta_noscale = np.zeros(X_train_noscale.shape[1])
theta_noscale, train_loss_history_noscale, valid_loss_history_noscale, train_acc_history_noscale, valid_acc_history_noscale = gradient_descent(
    X_train_noscale, y_train, X_val_noscale, y_val, theta_noscale, alpha, iterations
)

test_accuracy_minmax = compute_accuracy(X_test_minmax, y_test, theta_minmax)
test_accuracy_noscale = compute_accuracy(X_test_noscale, y_test, theta_noscale)

print(f"Test Accuracy (Min-Max Scaling): {test_accuracy_minmax}")
print(f"Test Accuracy (No Scaling): {test_accuracy_noscale}")

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(train_loss_history_minmax, label='Training Loss (Min-Max Scaling)', color='blue')
plt.plot(train_loss_history_noscale, label='Training Loss (No Scaling)', color='orange')
plt.title('Training Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(valid_loss_history_minmax, label='Validation Loss (Min-Max Scaling)', color='blue')
plt.plot(valid_loss_history_noscale, label='Validation Loss (No Scaling)', color='orange')
plt.title('Validation Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(train_acc_history_minmax, label='Training Accuracy (Min-Max Scaling)', color='blue')
plt.plot(train_acc_history_noscale, label='Training Accuracy (No Scaling)', color='orange')
plt.title('Training Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(valid_acc_history_minmax, label='Validation Accuracy (Min-Max Scaling)', color='blue')
plt.plot(valid_acc_history_noscale, label='Validation Accuracy (No Scaling)', color='orange')
plt.title('Validation Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
