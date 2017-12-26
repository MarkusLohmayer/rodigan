"""
This module contains functions that convert between Euler vectors,
unit quaternions and SO(3) matrices.
"""

import numba
import numpy as np


@numba.jit(numba.float64[:, :](numba.float64[:]), nopython=True, cache=True)
def matrix_from_quaternion(quaternion):
    """
    Computes the 3D rotation matrix that corresponds to the
    unit quaternion `q_real + i*q_i + j*q_j + k*q_k`

    Parameters
    ----------
    quaternion : array_like, shape = (4,), np.linalg.norm(q) equal to 1.0
        quaternion[0] = q_real
        quaternion[1] = q_i
        quaternion[2] = q_j
        quaternion[3] = q_k

    Returns
    -------
    matrix : ndarray, shape = (3, 3)
        Rotation matrix, SO(3)
    """
    if quaternion.shape != (4,):
        raise ValueError("quaternion is not of shape (4,)")

    if np.abs(np.linalg.norm(quaternion) - 1) > 0.0001:
        raise ValueError("quaternion is not a unit quaternion")

    if quaternion[0] < 0:
        raise ValueError("quaternion must have positive real part")

    matrix = np.zeros((3, 3), dtype=numba.float64)

    matrix[0, 0] = 1 - 2 * quaternion[2]**2 - 2 * quaternion[3]**2
    matrix[0, 1] = 2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3]
    matrix[0, 2] = 2 * quaternion[1] * quaternion[3] + 2 * quaternion[0] * quaternion[2]

    matrix[1, 0] = 2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3]
    matrix[1, 1] = 1 - 2 * quaternion[1]**2 - 2 * quaternion[3]**2
    matrix[1, 2] = 2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]

    matrix[2, 0] = 2 * quaternion[1] * quaternion[3] - 2 * quaternion[0] * quaternion[2]
    matrix[2, 1] = 2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1]
    matrix[2, 2] = 1 - 2 * quaternion[1]**2 - 2 * quaternion[2]**2

    return matrix



@numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, cache=True)
def quaternion_from_matrix(matrix):
    """
    Computes the unit quaternion that corresponds to a
    3D rotation matrix.

    Parameters
    ----------
    matrix : array_like, shape = (3, 3)
        Rotation matrix, SO(3)

    Returns
    -------
    quaternion : ndarray, shape = (4,)
        Unit quaternion with positive real part (first component).
    """
    if matrix.shape != (3, 3):
        raise ValueError("matrix is not of shape (3, 3)")

    quaternion = np.zeros((4,))

    # 4 cases for the sake of numerical stability
    cand = np.array([np.trace(matrix), matrix[0, 0], matrix[1, 1], matrix[2, 2]])
    index = np.argmax(cand)
    maxval = cand[index]

    if index == 0:
        quaternion[0] = np.sqrt(1 + maxval) * 0.5
        quaternion[1] = (matrix[2, 1] - matrix[1, 2]) / (4*quaternion[0])
        quaternion[2] = (matrix[0, 2] - matrix[2, 0]) / (4*quaternion[0])
        quaternion[3] = (matrix[1, 0] - matrix[0, 1]) / (4*quaternion[0])
    elif index == 1:
        quaternion[1] = np.sqrt(0.5*maxval + 0.25*(1 - np.trace(matrix)))
        quaternion[0] = (matrix[2, 1] - matrix[1, 2]) / (4*quaternion[1])
        quaternion[2] = (matrix[1, 0] + matrix[0, 1]) / (4*quaternion[1])
        quaternion[3] = (matrix[2, 0] + matrix[0, 2]) / (4*quaternion[1])
    elif index == 2:
        quaternion[2] = np.sqrt(0.5*maxval + 0.25*(1 - np.trace(matrix)))
        quaternion[0] = (matrix[0, 2] - matrix[2, 0]) / (4*quaternion[2])
        quaternion[1] = (matrix[0, 1] + matrix[1, 0]) / (4*quaternion[2])
        quaternion[3] = (matrix[2, 1] + matrix[1, 2]) / (4*quaternion[2])
    elif index == 3:
        quaternion[3] = np.sqrt(0.5*maxval + 0.25*(1 - np.trace(matrix)))
        quaternion[0] = (matrix[1, 0] - matrix[0, 1]) / (4*quaternion[3])
        quaternion[1] = (matrix[0, 2] + matrix[2, 0]) / (4*quaternion[3])
        quaternion[2] = (matrix[1, 2] + matrix[2, 1]) / (4*quaternion[3])

    # to make the representation unique, fix the real part to be positive
    if quaternion[0] < 0:
        quaternion *= -1

    assert np.abs(np.linalg.norm(quaternion) - 1) < 0.0001

    return quaternion



@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, cache=True)
def quaternion_from_euler(euler):
    """
    Computes the unit quaternion that corresponds to the
    axis-angle representation given by the (Euler) vector euler.
    The direction of the vector euler encodes the axis of rotation.
    The 2-norm of euler encodes the angle of rotation in radians.

    Note that unit quaternions can only represent rotations with
    0 <= theta <= 2pi, where theta it the Euler angle (norm of euler).

    Parameters
    ----------
    euler : array_like, shape = (3,)
        Euler vector (axis-angle representation)

    Returns
    -------
    quaternion : ndarray, shape = (4,)
        unit quaternion
    """
    if euler.shape != (3,):
        raise ValueError("euler is not of shape (3,)")

    # unit quaternion that corresponds to zero rotation
    quaternion = np.array([1, 0, 0, 0], dtype=numba.float64)

    theta = np.linalg.norm(euler)

    if theta < 1e-6:
        return quaternion

    axis = euler / theta
    quaternion[0] = np.cos(theta / 2)
    quaternion[1:] = np.sin(theta / 2) * axis

    if quaternion[0] < 0:
        quaternion *= -1

    return quaternion



@numba.jit(numba.float64[:](numba.float64[:]), nopython=True, cache=True)
def euler_from_quaternion(quaternion):
    """
    Computes the axis-angle representation that corresponds to the
    unit-quaternion representation of a 3D rotation.

    Parameters
    ----------
    quaternion : array_like, shape = (4,)
        Unit quaternion

    Returns
    -------
    euler : ndarray, shape = (3,)
        Euler vector (axis-angle representation)
    """
    if quaternion.shape != (4,):
        raise ValueError("quaternion is not of shape (4,)")

    if np.abs(np.linalg.norm(quaternion) - 1) > 0.0001:
        raise ValueError("quaternion is not a unit quaternion")

    if quaternion[0] < 0:
        raise ValueError("quaternion must have positive real part")

    euler = np.zeros((3), dtype=numba.float64)

    norm_of_q123 = np.linalg.norm(quaternion[1:])

    # probably this condition does not help precision
    if norm_of_q123 < 0.5:
        theta = 2 * np.arcsin(norm_of_q123)
    else:
        theta = 2 * np.arccos(quaternion[0])

    if theta > 1e-6:
        euler = (theta / norm_of_q123) * quaternion[1:]

    return euler




@numba.jit(numba.float64[:](numba.float64[:, :]), nopython=True, cache=True)
def euler_from_matrix(matrix):
    """
    Composition of the two functions `euler_from_quaternion` and
    `quaternion_from_matrix`.

    Parameters
    ----------
    matrix : ndarray , shape = (3, 3)
        Rotation matrix

    Returns
    -------
    euler : array_like , shape = (3,)
        Euler vector
    """
    return euler_from_quaternion(quaternion_from_matrix(matrix))



@numba.jit(numba.float64[:, :](numba.float64[:]), nopython=True, cache=True)
def matrix_from_euler(euler):
    """
    Composition of the two functions `matrix_from_quaternion` and
    `quaternion_from_euler`.

    Parameters
    ----------
    euler : array_like , shape = (3,)
        Euler vector

    Returns
    -------
    matrix : ndarray , shape = (3, 3)
        Rotation matrix
    """
    return matrix_from_quaternion(quaternion_from_euler(euler))
