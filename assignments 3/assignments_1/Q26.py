import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta, reg_type='none', reg_param=0.0):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  
    
    cost = (-1/m) * (y.dot(np.log(h + epsilon)) + (1 - y).dot(np.log(1 - h + epsilon)))
    
    if reg_type == 'l1':
        cost += reg_param * np.sum(np.abs(theta[1:])) / m
    elif reg_type == 'l2':
        cost += reg_param * np.sum(np.square(theta[1:])) / (2*m)
    
    return cost

def compute_accuracy(X, y, theta):
    predictions = sigmoid(X.dot(theta)) >= 0.5
    return np.mean(predictions == y)

def early_stopping(valid_loss_history, patience):
    if len(valid_loss_history) > patience:
        recent_losses = valid_loss_history[-patience:]
        if min(recent_losses) == valid_loss_history[-patience]:
            return True
    return False

def mini_batch_gradient_descent(X_train, y_train, X_valid, y_valid, theta, alpha, iterations, batch_size, patience, reg_type='none', reg_param=0.0):
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
        
        train_loss_history.append(compute_cost(X_train, y_train, theta, reg_type, reg_param))
        valid_loss_history.append(compute_cost(X_valid, y_valid, theta, reg_type, reg_param))
        train_acc_history.append(compute_accuracy(X_train, y_train, theta))
        valid_acc_history.append(compute_accuracy(X_valid, y_valid, theta))
        
        if early_stopping(valid_loss_history, patience):
            print(f"Early stopping at iteration {i}")
            break
        
        if i % 1000 == 0:
            print(f"Iteration {i}: Train Loss = {train_loss_history[-1]}, Valid Loss = {valid_loss_history[-1]}, Train Acc = {train_acc_history[-1]}, Valid Acc = {valid_acc_history[-1]}")

    return theta, train_loss_history, valid_loss_history, train_acc_history, valid_acc_history

np.random.seed(42)
X = np.random.rand(1000, 3)  
y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 3 > 2.5).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

theta = np.zeros(X_train.shape[1])
alpha = 0.001  
iterations = 2000
batch_size = 32
patience = 100  

regularization_types = ['none', 'l1', 'l2']
reg_params = [0, 0.01, 0.1, 1]
learning_rates = [0.001, 0.01, 0.1]

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

plt.figure(figsize=(18, 12))

for key, (train_loss, valid_loss, train_acc, valid_acc) in results.items():
    plt.plot(valid_loss, label=key)

plt.title('Validation Loss with Early Stopping (Different Learning Rates and Regularization)')
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()



