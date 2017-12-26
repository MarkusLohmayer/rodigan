"""
Unit tests for the kinematics module
"""

import numpy as np
import rotations
import kinematics


def test_update_euler():
    """
    Adding two known rotation matrices and check for the right result.
    """
    # rotate around z-axis by a random angle theta (-2pi < theta <+2pi)
    theta = np.pi/4

    known_matrix1 = np.array([[+np.cos(theta), -np.sin(theta), 0],
                              [+np.sin(theta), +np.cos(theta), 0],
                              [0, 0, 1]])
    euler1 = rotations.euler_from_matrix(known_matrix1)

    theta = np.pi/2

    known_matrix2 = np.array([[+np.cos(theta), -np.sin(theta), 0],
                              [+np.sin(theta), +np.cos(theta), 0],
                              [0, 0, 1]])
    euler2 = rotations.euler_from_matrix(known_matrix2)

    theta = np.pi/4 + np.pi/2

    known_matrix3 = np.array([[+np.cos(theta), -np.sin(theta), 0],
                              [+np.sin(theta), +np.cos(theta), 0],
                              [0, 0, 1]])
    euler3 = rotations.euler_from_matrix(known_matrix3)

    assert np.allclose(kinematics.update_euler(euler1, euler2), euler3)


def random_euler_and_matrix():
    """
    Returns
    -------
    euler : ndarray , shape = (3,)
        Random Euler vector with norm smaller than pi
    matrix : ndarray , shape = (3, 3)
        Corresponding SO(3) matrix
    """
    # create a random rotation in axis-angle representation
    while True:
        euler = np.random.normal(loc=0.0, scale=1.0, size=3)
        if np.linalg.norm(euler) < 1*np.pi:
            break

    # convert to a matrix
    matrix = rotations.matrix_from_quaternion(rotations.quaternion_from_euler(euler))

    assert np.allclose(np.dot(matrix, matrix.T), np.eye(3))

    return euler, matrix


def euler_and_matrix_from_theta(theta):
    """
    Parameters
    ----------
    theta : float
        Angle of rotation about the e_3 axis given in radians

    Returns
    -------
    euler : ndarray , shape = (3,)
        Euler vector: e_3 * theta
    matrix : ndarray , shape = (3, 3)
        Rotation matrix about the e_3 axis with given angle.
    """
    if not isinstance(theta, float):
        try:
            theta = float(theta)
        except:
            raise TypeError('theta must be a floating point number!')
    euler = np.array([0, 0, theta])
    matrix = np.array([[+np.cos(theta), -np.sin(theta), 0],
                       [+np.sin(theta), +np.cos(theta), 0],
                       [0, 0, 1]])
    return euler, matrix


def test_interpolate_euler():
    """
    Generates two random Euler vectors and their corresponding SO(3) matrices.
    Then
    """
    theta1 = np.random.uniform(0, np.pi, size=1)
    euler1, _ = euler_and_matrix_from_theta(theta1)

    theta2 = np.random.uniform(0, np.pi, size=1)
    euler2, _ = euler_and_matrix_from_theta(theta2)

    # Interpolating matrix
    matrix3_test = kinematics.interpolate_euler(euler1, euler2)

    # this is allowed when axes are alligned
    theta3 = (theta1 + theta2) / 2
    euler3 = np.array([0, 0, theta3])
    matrix3 = rotations.matrix_from_euler(euler3)

    assert np.allclose(matrix3, matrix3_test)
