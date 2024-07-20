import numpy as np


def test(result, expected, tol=1e-4):
    try:
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        np.testing.assert_allclose(result, expected, atol=tol)
        print("Test passed!")
    except AssertionError as msg:
        print(f"Test failed: {msg}")
