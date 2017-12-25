"""
Unit tests for the rotations module
"""


import numpy as np
import auxiliary


def test_cross_products():
    """
    Test if the two functions `cross` and `skew_matrix_from_vector` produce
    the same output as np.cross.
    """
    vector1 = np.random.normal(loc=0.0, scale=10.0, size=3)
    vector2 = np.random.normal(loc=0.0, scale=10.0, size=3)

    assert np.allclose(np.cross(vector1, vector2),
                       auxiliary.cross(vector1, vector2))

    assert np.allclose(np.cross(vector1, vector2),
                       np.dot(auxiliary.skew_matrix_from_vector(vector1), vector2))
