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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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

        # Save the cost and accuracy for each iteration
        train_loss_history[i] = compute_cost(X_train, y_train, theta)
        valid_loss_history[i] = compute_cost(X_valid, y_valid, theta)
        train_acc_history[i] = compute_accuracy(X_train, y_train, theta)
        valid_acc_history[i] = compute_accuracy(X_valid, y_valid, theta)

        if i % 1000 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss_history[i]}, Valid Loss = {valid_loss_history[i]}, Train Acc = {train_acc_history[i]}, Valid Acc = {valid_acc_history[i]}")

    return theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history

def stochastic_gradient_descent(X_train, y_train, X_valid, y_valid, theta, alpha, iterations):
    m = len(y_train)
    train_loss_history = np.zeros(iterations)
    valid_loss_history = np.zeros(iterations)
    train_acc_history = np.zeros(iterations)
    valid_acc_history = np.zeros(iterations)
    
    for i in range(iterations):
        for j in range(m):
            idx = np.random.randint(m)
            xi = X_train[idx:idx+1]
            yi = y_train[idx:idx+1]
            gradient = xi.T.dot(sigmoid(xi.dot(theta)) - yi)
            theta -= alpha * gradient
        
        train_loss_history[i] = compute_cost(X_train, y_train, theta)
        valid_loss_history[i] = compute_cost(X_valid, y_valid, theta)
        train_acc_history[i] = compute_accuracy(X_train, y_train, theta)
        valid_acc_history[i] = compute_accuracy(X_valid, y_valid, theta)

        if i % 100 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss_history[i]}, Valid Loss = {valid_loss_history[i]}, Train Acc = {train_acc_history[i]}, Valid Acc = {valid_acc_history[i]}")

    return theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history

def mini_batch_gradient_descent(X_train, y_train, X_valid, y_valid, theta, alpha, iterations, batch_size):
    m = len(y_train)
    train_loss_history = np.zeros(iterations)
    valid_loss_history = np.zeros(iterations)
    train_acc_history = np.zeros(iterations)
    valid_acc_history = np.zeros(iterations)
    
    for i in range(iterations):
        indices = np.random.permutation(m)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for j in range(0, m, batch_size):
            end = min(j + batch_size, m)
            X_batch = X_train_shuffled[j:end]
            y_batch = y_train_shuffled[j:end]
            gradient = (1 / len(y_batch)) * X_batch.T.dot(sigmoid(X_batch.dot(theta)) - y_batch)
            theta -= alpha * gradient
        
        train_loss_history[i] = compute_cost(X_train, y_train, theta)
        valid_loss_history[i] = compute_cost(X_valid, y_valid, theta)
        train_acc_history[i] = compute_accuracy(X_train, y_train, theta)
        valid_acc_history[i] = compute_accuracy(X_valid, y_valid, theta)

        if i % 100 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss_history[i]}, Valid Loss = {valid_loss_history[i]}, Train Acc = {train_acc_history[i]}, Valid Acc = {valid_acc_history[i]}")

    return theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history

theta = np.zeros(X_train.shape[1])
theta_sgd = np.zeros(X_train.shape[1])
theta_mini_batch_32 = np.zeros(X_train.shape[1])
theta_mini_batch_64 = np.zeros(X_train.shape[1])
alpha = 0.001 
iterations = 1000

theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history = gradient_descent(
    X_train, y_train, X_val, y_val, theta, alpha, iterations
)
theta_sgd, train_loss_sgd, valid_loss_sgd, train_acc_sgd, valid_acc_sgd = stochastic_gradient_descent(
    X_train, y_train, X_val, y_val, theta_sgd, alpha, iterations
)

theta_mini_batch_32, train_loss_mini_batch_32, valid_loss_mini_batch_32, train_acc_mini_batch_32, valid_acc_mini_batch_32 = mini_batch_gradient_descent(
    X_train, y_train, X_val, y_val, theta_mini_batch_32, alpha, iterations, batch_size=32
)

theta_mini_batch_64, train_loss_mini_batch_64, valid_loss_mini_batch_64, train_acc_mini_batch_64, valid_acc_mini_batch_64 = mini_batch_gradient_descent(
    X_train, y_train, X_val, y_val, theta_mini_batch_64, alpha, iterations, batch_size=64
)

test_accuracy = compute_accuracy(X_test, y_test, theta)
print(f"Test Accuracy for Gradient Descent: {test_accuracy}")

test_accuracy_sgd = compute_accuracy(X_test, y_test, theta_sgd)
print(f"Test Accuracy for SGD: {test_accuracy_sgd}")

test_accuracy_mini_batch_32 = compute_accuracy(X_test, y_test, theta_mini_batch_32)
print(f"Test Accuracy for Mini-Batch 32: {test_accuracy_mini_batch_32}")

test_accuracy_mini_batch_64 = compute_accuracy(X_test, y_test, theta_mini_batch_64)
print(f"Test Accuracy for Mini-Batch 64: {test_accuracy_mini_batch_64}")

plt.figure(figsize=(18, 12))

plt.subplot(3, 4, 1)
plt.plot(train_loss_history, label='Gradient Descent', color='blue')
plt.plot(train_loss_sgd, label='SGD', color='red')
plt.plot(train_loss_mini_batch_32, label='Mini-Batch (32)', color='green')
plt.plot(train_loss_mini_batch_64, label='Mini-Batch (64)', color='orange')
plt.title('Training Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 4, 2)
plt.plot(valid_loss_history, label='Gradient Descent', color='blue')
plt.plot(valid_loss_sgd, label='SGD', color='red')
plt.plot(valid_loss_mini_batch_32, label='Mini-Batch (32)', color='green')
plt.plot(valid_loss_mini_batch_64, label='Mini-Batch (64)', color='orange')
plt.title('Validation Loss vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 4, 3)
plt.plot(train_acc_history, label='Gradient Descent', color='blue')
plt.plot(train_acc_sgd, label='SGD', color='red')
plt.plot(train_acc_mini_batch_32, label='Mini-Batch (32)', color='green')
plt.plot(train_acc_mini_batch_64, label='Mini-Batch (64)', color='orange')
plt.title('Training Accuracy vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(3, 4, 4)
plt.plot(valid_acc_history, label='Gradient Descent', color='blue')
plt.plot(valid_acc_sgd, label='SGD', color='red')
plt.plot(valid_acc_mini_batch_32, label='Mini-Batch (32)', color='green')
plt.plot(valid_acc_mini_batch_64, label='Mini-Batch (64)', color='orange')
plt.title('Validation Accuracy vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
