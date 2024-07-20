"""
Write a Python function that takes the dot product of a matrix and a vector. return -1 if the matrix could not be dotted with the vector

Example:
        input: a = [[1,2],[2,4]], b = [1,2]
        output:[5, 10] 
        reasoning: 1*1 + 2*2 = 5;
                   1*2+ 2*4 = 10
"""

from utils.test import test
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def matrix_times_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float]:
    # matrix can not be dotted with vector due to unappropiate dimension
    if len(b) != len(a[0]):
        return -1

    doc_product_result = []

    for row in a:
        acc = 0
        for i in range(0, len(b)):
            acc += row[i] * b[i]

        doc_product_result.append(acc)

    return doc_product_result


a = [[1, 2], [2, 4]]
b = [1, 2]
expected = [5, 10]
test(matrix_times_vector(a, b), expected)

a = [[1, 2, 3], [2, 4, 5], [6, 8, 9]]
b = [1, 2, 3]
expected = [14, 25, 49]
test(matrix_times_vector(a, b), expected)

a = [[1, 2], [2, 4], [6, 8], [12, 4]]
b = [1, 2, 3]
expected = -1
test(matrix_times_vector(a, b), expected)
