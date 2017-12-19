""" Auxiliary functions """

import numpy as np

import numba

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@numba.jit(numba.float64[:, :](numba.float64[:]), nopython=True)
def skew_matrix_from_vector(axial_vector):
    """
    Computes the skew symmetric matrix corresponding to the axial vector

    Parameters
    ----------
    axial_vector : array_like, shape = (3,)

    Returns
    -------
    skew_matrix : ndarray, shape = (3, 3)
    """
    assert axial_vector.shape == (3,)

    skew_matrix = np.zeros((3, 3), dtype=numba.float64)

    skew_matrix[0, 1] = -axial_vector[2]
    skew_matrix[0, 2] = +axial_vector[1]

    skew_matrix[1, 0] = +axial_vector[2]
    skew_matrix[1, 2] = -axial_vector[0]

    skew_matrix[2, 0] = -axial_vector[1]
    skew_matrix[2, 1] = +axial_vector[0]

    return skew_matrix


@numba.jit(numba.float64[:, :](numba.float64[:]), nopython=True)
def rotation_matrix_from_quaternion(quaternion):
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
    rotation_matrix : ndarray, shape = (3, 3)
    """
    assert quaternion.shape == (4,)
    assert np.abs(np.linalg.norm(quaternion) - 1) < 0.0001

    rotation_matrix = np.zeros((3, 3), dtype=numba.float64)

    rotation_matrix[0, 0] = 1 - 2 * quaternion[2]**2 - 2 * quaternion[3]**2
    rotation_matrix[0, 1] = 2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3]
    rotation_matrix[0, 2] = 2 * quaternion[1] * quaternion[3] + 2 * quaternion[0] * quaternion[2]

    rotation_matrix[1, 0] = 2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3]
    rotation_matrix[1, 1] = 1 - 2 * quaternion[1]**2 - 2 * quaternion[3]**2
    rotation_matrix[1, 2] = 2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]

    rotation_matrix[2, 0] = 2 * quaternion[1] * quaternion[3] - 2 * quaternion[0] * quaternion[2]
    rotation_matrix[2, 1] = 2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1]
    rotation_matrix[2, 2] = 1 - 2 * quaternion[1]**2 - 2 * quaternion[2]**2

    return rotation_matrix




@numba.jit(numba.float64[:, :](numba.float64[:]), nopython=True)
def rotation_matrix_from_axis_anlge(axis_angle):
    """
    Computes the 3D rotaion matrix that corresponds to the
    axis-angle representation given by the vector axis_angle.
    The direction of axis_angle encodes the axis of rotation.
    The 2-norm of axis_angle encodes the angle of rotation in radians.

    The function first converts the axis-angle representation into the
    correponding unit quaternion and then uses the function
    rotation_matrix_from_quaternion from the same module to
    compute the rotation matrix.

    Parameters
    ----------
    axis_angle : array_like, shape = (3,)

    Returns
    -------
    rotation_matrix : ndarray, shape = (3, 3)
    """
    assert axis_angle.shape == (3,)

    theta = np.linalg.norm(axis_angle)

    if theta < 1e-6:
        rotation_matrix = np.zeros((3, 3), dtype=numba.float64)
        for i in range(3):
            rotation_matrix[i, i] = 1
        return rotation_matrix

    axis = axis_angle / theta
    unit_quaternion = np.zeros((4,), dtype=numba.float64)
    unit_quaternion[0] = np.cos(theta / 2)
    unit_quaternion[1:] = np.sin(theta / 2) * axis
    return rotation_matrix_from_quaternion(unit_quaternion)


def test_rotation_matrix_from_axis_anlge():
    """
    Tests the function `rotation_matrix_from_axis_anlge` by creating
    a known rotation matrix and comparing the results.
    """
    # rotate around z-axis by angle theta
    theta = (3/2)*np.pi
    known_rotation_matrix = np.array([[+np.cos(theta), -np.sin(theta), 0],
                                      [+np.sin(theta), +np.cos(theta), 0],
                                      [0, 0, 1]])
    axis_angle = theta * np.array([0, 0, 1])
    rotation_marrix = rotation_matrix_from_axis_anlge(axis_angle)
    assert np.allclose(known_rotation_matrix, rotation_marrix)

    # further, assert that the rotation matrix is unitary
    assert np.allclose(np.eye(3), np.dot(rotation_matrix.T, rotation_matrix))


def quaternion_from_rotation_matrix(rotation_matrix):
    """
    Computes the unit quaternion that corresponds to the
    3D rotation matrix.

    Parameters
    ----------
    rotation_matrix : ndarray, shape = (3, 3)

    Returns
    -------
    unit_quaternion : ndarray, shape = (4,)
    """
    assert rotation_marrix.shape == (3, 3)

    unit_quaternion = np.zeros((4,))

    cand = np.array([np.trace(rotation_matrix), rotation_matrix[0,0], rotation_matrix[1,1], rotation_matrix[2,2]])
    ind = np.argmax(cand)
    maxval = cand[ind]

    if ind ==0:
        unit_quaternion[0] = np.sqrt(1 + maxval) * 0.5
        unit_quaternion[1] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4*unit_quaternion[0])
        unit_quaternion[2] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4*unit_quaternion[0])
        unit_quaternion[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4*unit_quaternion[0])
    elif ind==1:
        unit_quaternion[1] = np.sqrt(0.5*maxval + 0.25*(1 - np.trace(rotation_matrix)))
        unit_quaternion[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4*unit_quaternion[1])
        unit_quaternion[2] = (rotation_matrix[1, 0] + rotation_matrix[0, 1]) / (4*unit_quaternion[1])
        unit_quaternion[3] = (rotation_matrix[2, 0] + rotation_matrix[0, 2]) / (4*unit_quaternion[1])
    elif ind==2:
        unit_quaternion[2] = np.sqrt(0.5*maxval + 0.25*(1 - np.trace(rotation_matrix)))
        unit_quaternion[0] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4*unit_quaternion[2])
        unit_quaternion[1] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / (4*unit_quaternion[2])
        unit_quaternion[3] = (rotation_matrix[2, 1] + rotation_matrix[1, 2]) / (4*unit_quaternion[2])
    elif ind==3:
        unit_quaternion[3] = np.sqrt(0.5*maxval + 0.25*(1 - np.trace(rotation_matrix)))
        unit_quaternion[0] = (rotation_matrix[1,0] - rotation_matrix[0,1]) / (4*unit_quaternion[3])
        unit_quaternion[1] = (rotation_matrix[0,2] + rotation_matrix[2,0]) / (4*unit_quaternion[3])
        unit_quaternion[2] = (rotation_matrix[1,2] + rotation_matrix[2,1]) / (4*unit_quaternion[3])
    else:
        raise Exception

    assert np.abs(np.linalg.norm(unit_quaternion) - 1) < 0.0001

    return unit_quaternion


def plot_centerline(centerline):
    """
    Create a 3D plot of the beam's centerline

    Parameters
    ----------
    centerline : array_like
        shape[0] = 3 (dimensions of space)
        spape[1] = n_n (number of nodes)
    """
    assert centerline.shape[0] == 3
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot(centerline[0, :], centerline[1, :], centerline[2, :])
