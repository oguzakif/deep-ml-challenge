"""
Write a Python function that calculates the covariance matrix from a list of vectors. 
Assume that the input list represents a dataset where each vector is a feature, and vectors are of equal length.

Example:
        input: vectors = [[1, 2, 3], [4, 5, 6]]
        output: [[1.0, 1.0], [1.0, 1.0]]
        reasoning: The dataset has two features with three observations each. 
        The covariance between each pair of features (including covariance with itself) is calculated and returned as a 2x2 matrix.
"""

from utils.test import test
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    feature_count = len(vectors)
    feature_observation_count = len(vectors[0])
    covariance_matrix = [
        [0 for _ in range(feature_count)] for _ in range(feature_count)
    ]
    for i in range(feature_count):
        vector_mean = calculate_mean(vectors[i])
        for j in range(i, feature_count):
            covariance = calculate_covariance(
                vectors[i], vectors[j], vector_mean, feature_observation_count
            )
            covariance_matrix[i][j] = covariance
            covariance_matrix[j][i] = covariance

    return covariance_matrix


def calculate_covariance(
    vector1: list[float],
    vector2: list[float],
    vector_mean: float,
    feature_example_count: int,
) -> float:
    return sum(
        (vector1[k] - vector_mean) * (vector2[k] - vector_mean)
        for k in range(feature_example_count)
    ) / (feature_example_count - 1)


def calculate_mean(vector: list[float]) -> float:
    acc = 0
    for val in vector:
        acc += val

    return acc / len(vector)


vectors = [[1, 2, 3], [4, 5, 6]]
expected = [[1.0, 1.0], [1.0, 1.0]]
test(calculate_covariance_matrix(vectors), expected)


vectors = [[1, 5, 6], [2, 3, 4], [7, 8, 9]]
expected = [[7.0, 2.5, 2.5], [2.5, 1.0, 1.0], [2.5, 1.0, 1.0]]
test(calculate_covariance_matrix(vectors), expected)
