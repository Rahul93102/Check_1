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

X = np.c_[np.ones(X.shape[0]), X]  #

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])

def precision_score(conf_matrix):
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall_score(conf_matrix):
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
def gradient_descent(X_train, y_train, X_val, y_val, theta, alpha, iterations):
    m = len(y_train)
    for i in range(iterations):
        gradient = (1/m) * X_train.T.dot(sigmoid(X_train.dot(theta)) - y_train)
        theta -= alpha * gradient
    return theta

def k_fold_cross_validation(X, y, k=5, alpha=0.001, iterations=1000):
    fold_size = len(X) // k
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for fold in range(k):
        X_val = X[fold*fold_size:(fold+1)*fold_size]
        y_val = y[fold*fold_size:(fold+1)*fold_size]
        X_train = np.concatenate([X[:fold*fold_size], X[(fold+1)*fold_size:]], axis=0)
        y_train = np.concatenate([y[:fold*fold_size], y[(fold+1)*fold_size:]], axis=0)
        
        theta = np.zeros(X_train.shape[1])

        theta = gradient_descent(X_train, y_train, X_val, y_val, theta, alpha, iterations)

        y_val_pred = sigmoid(X_val.dot(theta)) >= 0.5

        conf_matrix = confusion_matrix(y_val, y_val_pred)
        accuracy = compute_accuracy(X_val, y_val, theta)
        precision = precision_score(conf_matrix)
        recall = recall_score(conf_matrix)
        f1 = f1_score(precision, recall)

        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

        print(f"Fold {fold + 1}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1}")

    return metrics

k = 5
metrics = k_fold_cross_validation(X_train, y_train, k=k)

for metric in metrics:
    avg = np.mean(metrics[metric])
    std = np.std(metrics[metric])
    print(f"{metric.capitalize()} - Average: {avg}, Standard Deviation: {std}")

theta = np.zeros(X_train.shape[1])
theta = gradient_descent(X_train, y_train, X_test, y_test, theta, alpha=0.001, iterations=1000)
y_test_pred = sigmoid(X_test.dot(theta)) >= 0.5
test_accuracy = compute_accuracy(X_test, y_test, theta)

conf_matrix = confusion_matrix(y_test, y_test_pred)
precision = precision_score(conf_matrix)
recall = recall_score(conf_matrix)
f1 = f1_score(precision, recall)

print("\nTest Set Performance:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
