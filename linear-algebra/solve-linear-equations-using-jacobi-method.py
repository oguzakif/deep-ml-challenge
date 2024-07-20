"""
Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. 
The function should iterate 10 times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

Example:
        input: A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
        output: [0.146, 0.2032, -0.5175]
        reasoning: The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), 
        where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.
"""

from utils.test import test
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    diagonal_vals = np.diag(A)
    non_diagonal_val = A - np.diag(diagonal_vals)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1 / diagonal_vals[i]) * (b[i] - sum(non_diagonal_val[i] * x))
        x = x_hold.copy()
    return np.round(x, 4).tolist()


A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
b = [-1, 2, 3]
n = 2
expected = [0.146, 0.2032, -0.5175]
test(solve_jacobi(A, b, n), expected)


A = [[4, 1, 2], [1, 5, 1], [2, 1, 3]]
b = [4, 6, 7]
n = 5
expected = [-0.0806, 0.9324, 2.4422]
test(solve_jacobi(A, b, n), expected)

A = [[4, 2, -2], [1, -3, -1], [3, -1, 4]]
b = [0, 7, 5]
n = 3
expected = [1.7083, -1.9583, -0.7812]
test(solve_jacobi(A, b, n), expected)
