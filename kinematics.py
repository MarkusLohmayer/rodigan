"""
Geometric properties and kinematics of the special Cosserat rod.
"""

import numba
import numpy as np

import rotations


class Geometry:
    """
    This class holds the geometry of the rod.
    """
    def __init__(self, length, radius):
        """
        Create a new geometry by setting the length and radius of the rod.
        """
        self.length = length
        self.radius = radius


    @property
    def length(self):
        """Length of the rod."""
        return self.__length


    @length.setter
    def length(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError('The length must be given as a positive number!')
        if value <= 0:
            raise ValueError('The length must be positive!')
        # pylint: disable=W0201
        self.__length = float(value)


    @property
    def radius(self):
        """The rod's cross-section radius."""
        return self.__radius


    @radius.setter
    def radius(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError('The radius must be given as a positive number!')
        if value <= 0:
            raise ValueError('The radius must be positive!')
        if value >= self.length / 10:
            raise ValueError('The radius is too big. Rods are by definition slender.')
        # pylint: disable=W0201
        self.__radius = float(value)


    @property
    def cross_section_area(self):
        """The rod's cross-section area calculated based on the given radius."""
        return np.pi * self.radius**2


    @property
    def second_moment_of_area(self):
        """The rod's cross-section second moment of area calculated based on the given radius."""
        return np.pi * self.radius**4 / 4





@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:]), nopython=True, cache=True)
def update_euler(current_euler, increment_euler):
    """
    Function for composing two rotations given in axis-angle representation.
    The function first converts both rotations to their matrix representation,
    then forms the matrix product increment*current and converts back
    to axis-angle representation.

    Parameters
    ----------
    current_euler : array_like , shape = (3,)
    increment_euler : array_like , shape = (3,)

    Returns
    -------
    updated_euler : ndarray , shape = (3,)
    """
    # convert to SO(3) matrices
    increment_matrix = rotations.matrix_from_euler(increment_euler)
    current_matrix = rotations.matrix_from_euler(current_euler)
    # compute resulting SO(3) matrix
    updated_matrix = np.dot(increment_matrix, current_matrix)
    # convert back to Euler vector
    updated_euler = rotations.euler_from_matrix(updated_matrix)
    return updated_euler



@numba.jit(numba.float64[:, :](numba.float64[:], numba.float64[:]), nopython=True, cache=True)
def interpolate_euler(euler_left, euler_right):
    """
    Compute the rotation matrix that interpolates between the two given
    Euler vectors `euler_left` and `euler_right` in the following sense:

    matrix_right = matrix_difference_halved * matrix_difference_halved * matrix_left
    matrix_interpolant = matrix_difference_halved * matrix_left

    Parameters
    ----------
    euler_left : array_like , shape = (3,)
        Euler vector that corresponds to matrix_left
    euler_right : array_like , shape = (3,)
        Euler vector that corresponds to matrix_right

    Returns
    -------
    matrix_interpolant : ndarray , shape = (3, 3)
        Interpolating rotation matrix
    """
    # convert to SO(3) matrices
    matrix_left = rotations.matrix_from_euler(euler_left)
    matrix_right = rotations.matrix_from_euler(euler_right)
    # compute the "difference" rotation matrix, that transforms left into right
    matrix_difference = np.dot(matrix_right, matrix_left.T)
    euler_difference = rotations.euler_from_matrix(matrix_difference)
    # split this "difference" rotation into two identical rotations,
    # that give the same result when multiplied together
    euler_difference_halved = euler_difference / 2
    matrix_difference_halved = rotations.matrix_from_euler(euler_difference_halved)
    # "add" this rotation to the left node's rotation matrix
    matrix_interpolant = np.dot(matrix_difference_halved, matrix_left)
    return matrix_interpolant
