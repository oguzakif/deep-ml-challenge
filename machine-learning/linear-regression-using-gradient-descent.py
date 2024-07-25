"""
Write a Python function that performs linear regression using gradient descent. 
The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, 
along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. 
Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

Example:
        input: X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
        output: np.array([0.1107, 0.9513])
        reasoning: The linear model is y = 0.0 + 1.0*x, which fits the input data after gradient descent optimization.
"""

import numpy as np
from utils.test import test
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))

    for i in range(iterations):
        predictions = predict(X, theta)
        errors = calculate_errors(predictions, y.reshape(-1, 1))  # calculate errors
        updates = calculate_updates(X.T, errors, m)  # calculate updates
        theta -= alpha * updates  # multiply learning rate with weight updates
    theta = np.round(theta, 4).flatten().tolist()  # round to 4 digits

    return theta


def calculate_errors(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return predicted - actual


def calculate_updates(
    feature_transpose: np.ndarray, errors: np.ndarray, training_example_count: int
) -> np.ndarray:
    return feature_transpose @ errors / training_example_count


def predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features @ weights


X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
alpha = 0.01
iterations = 1000
expected = np.array([0.1107, 0.9513])
test(linear_regression_gradient_descent(X, y, alpha, iterations), expected)


X = np.array([[1, 1, 3], [1, 2, 4], [1, 3, 5]])
y = np.array([2, 3, 5])
alpha = 0.1
iterations = 10
expected = np.array([-1.0241, -1.9133, -3.9616])
test(linear_regression_gradient_descent(X, y, alpha, iterations), expected)
