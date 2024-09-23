import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./assignments_1/HeartDisease.csv')

data.fillna(data.mean(), inplace=True)

X = data.drop("HeartDisease", axis=1).values  # Features (all columns except HeartDisease)
y = data["HeartDisease"].values  # Target (HeartDisease column)

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

from sklearn.model_selection import train_test_split
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

def auc_score(y_true, y_prob):
    thresholds = np.arange(0, 1.1, 0.1)
    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = y_prob >= threshold
        conf_matrix = confusion_matrix(y_true, y_pred)
        TP = conf_matrix[1, 1]
        FP = conf_matrix[0, 1]
        TN = conf_matrix[0, 0]
        FN = conf_matrix[1, 0]
        tpr.append(TP / (TP + FN))
        fpr.append(FP / (FP + TN))
    return np.trapz(tpr, fpr)

y_test_prob = sigmoid(X_test.dot(theta))
y_test_pred = y_test_prob >= 0.5

conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(conf_matrix)
recall = recall_score(conf_matrix)
f1 = f1_score(precision, recall)
auc = auc_score(y_test, y_test_prob)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC: {auc}")
