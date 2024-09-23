import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('./assignments_1/HeartDisease.csv')

data.fillna(data.mean(), inplace=True)

X = data.drop("HeartDisease", axis=1).values  
y = data["HeartDisease"].values  

std_dev = X.std(axis=0)
constant_columns = np.where(std_dev == 0)[0]
if len(constant_columns) > 0:
    print(f"Constant columns detected: {constant_columns}")
    X = np.delete(X, constant_columns, axis=1)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  
X = (X - X_mean) / X_std

X = np.c_[np.ones(X.shape[0]), X]  

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # To avoid log(0)
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

theta = np.zeros(X_train.shape[1])
alpha = 0.001  
iterations = 30000

theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history = gradient_descent(
    X_train, y_train, X_val, y_val, theta, alpha, iterations
)

test_accuracy = compute_accuracy(X_test, y_test, theta)
print(f"Test Accuracy: {test_accuracy}")

plt.figure(figsize=(14, 10))

# Plot Training Loss vs. Iteration
plt.subplot(2, 2, 1)
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.title('Training Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss vs. Iteration
plt.subplot(2, 2, 2)
plt.plot(valid_loss_history, label='Validation Loss', color='red')
plt.title('Validation Loss vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

# Plot Training Accuracy vs. Iteration
plt.subplot(2, 2, 3)
plt.plot(train_acc_history, label='Training Accuracy', color='green')
plt.title('Training Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()

# Plot Validation Accuracy vs. Iteration
plt.subplot(2, 2, 4)
plt.plot(valid_acc_history, label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
