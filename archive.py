import numpy as np

#numpy version, not compatible with numba
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
    return np.array([[0, -axial_vector[2], axial_vector[1]],
                     [axial_vector[2], 0, -axial_vector[0]],
                     [-axial_vector[1], axial_vector[0], 0]])


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
    assert np.allclose(np.linalg.norm(quaternion), 1)
    return 2 * np.array([[0.5 - quaternion[2]**2 - quaternion[3]**2,
                          quaternion[1] * quaternion[2] - quaternion[0] * quaternion[3],
                          quaternion[1] * quaternion[3] + quaternion[0] * quaternion[2]],
                         [quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3],
                          0.5 - quaternion[1]**2 - quaternion[3]**2,
                          quaternion[2] * quaternion[3] - quaternion[0] * quaternion[1]],
                         [quaternion[1] * quaternion[3] - quaternion[0] * quaternion[2],
                          quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1],
                          0.5 - quaternion[1]**2 - quaternion[2]**2]])


def quaternion_from_rotation_matrix2(rotation_matrix):
    """
    Computes the unit quaternion that corresponds to the
    3D rotation matrix.

    The algorith is based on the article
    "Converting a Rotation Matrix to a Quaternion" by Mike Day.

    Parameters
    ----------
    rotation_matrix : ndarray, shape = (3, 3)

    Returns
    -------
    unit_quaternion : ndarray, shape = (4,)
    """
    assert rotation_matrix.shape == (3, 3)

    unit_quaternion = np.zeros((4,))

    if rotation_matrix[2, 2] < 0:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1]:
            t = 1 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]
            unit_quaternion[0] = t
            unit_quaternion[1] = rotation_matrix[0, 1] + rotation_matrix[1, 0]
            unit_quaternion[2] = rotation_matrix[2, 0] + rotation_matrix[0, 2]
            unit_quaternion[3] = rotation_matrix[1, 2] - rotation_matrix[2, 1]
        else:
            t = 1 - rotation_matrix[0, 0] + rotation_matrix[1, 1] - rotation_matrix[2, 2]
            unit_quaternion[0] = rotation_matrix[0, 1] + rotation_matrix[1, 0]
            unit_quaternion[1] = t
            unit_quaternion[2] = rotation_matrix[1, 2] + rotation_matrix[2, 1]
            unit_quaternion[3] = rotation_matrix[2, 0] - rotation_matrix[0, 2]
    else:
        if rotation_matrix[0, 0] < -rotation_matrix[1, 1]:
            t = 1 - rotation_matrix[0, 0] - rotation_matrix[1, 1] + rotation_matrix[2, 2]
            unit_quaternion[0] = rotation_matrix[2, 0] + rotation_matrix[0, 2]
            unit_quaternion[1] = rotation_matrix[1, 2] + rotation_matrix[2, 1]
            unit_quaternion[2] = t
            unit_quaternion[3] = rotation_matrix[0, 1] - rotation_matrix[1, 0]
        else:
            t = 1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
            unit_quaternion[0] = rotation_matrix[1, 2] - rotation_matrix[2, 1]
            unit_quaternion[1] = rotation_matrix[2, 0] - rotation_matrix[0, 2]
            unit_quaternion[2] = rotation_matrix[0, 1] - rotation_matrix[1, 0]
            unit_quaternion[3] = t

    unit_quaternion *= 0.5 / np.sqrt(t)

    assert np.abs(np.linalg.norm(unit_quaternion) - 1) < 0.0001

    return unit_quaternion
