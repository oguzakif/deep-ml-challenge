"""
Write a Python function that performs linear regression using the normal equation. 
The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. 
Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

Example:
        input: X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
        output: [0.0, 1.0]
        reasoning: The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.
"""

import numpy as np
from utils.test import test
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def linear_regression_normal_equation(
    X: list[list[float]], y: list[float]
) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # convert y to column vector
    X_transpose = X.T  # get X transpose

    # θ=((XT.X)^−1).XT.y
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    theta = np.round(theta, 4).flatten().tolist()  # round to 4 digits
    return theta


X = [[1, 1], [1, 2], [1, 3]]
y = [1, 2, 3]
expected = [-0.0, 1.0]
test(linear_regression_normal_equation(X, y), expected)


X = [[1, 3, 4], [1, 2, 5], [1, 3, 2]]
y = [1, 2, 1]
expected = [4.0, -1.0, -0.0]
test(linear_regression_normal_equation(X, y), expected)
