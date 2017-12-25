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
    if axial_vector.shape != (3,):
        raise ValueError("axial_vector is not of shape (3,)")

    skew_matrix = np.zeros((3, 3), dtype=numba.float64)

    skew_matrix[0, 1] = -axial_vector[2]
    skew_matrix[0, 2] = +axial_vector[1]

    skew_matrix[1, 0] = +axial_vector[2]
    skew_matrix[1, 2] = -axial_vector[0]

    skew_matrix[2, 0] = -axial_vector[1]
    skew_matrix[2, 1] = +axial_vector[0]

    return skew_matrix



@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:]), nopython=True)
def cross(vector1, vector2):
    """
    Calculates the cross product of two 3D vectors. This function was written
    because it seems that `numpy.cross` is not supported by `numba.jit`.

    Parameters
    ----------
    vector1 : array_like, shape = (3,)
    vector2 : array_like, shape = (3,)

    Returns
    -------
    cross_product : ndarray, shape = (3,)
        Cross product vector1 x vector2
    """
    cross_product = np.zeros((3), dtype=numba.float64)
    cross_product[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    cross_product[1] = vector1[2] * vector2[0] - vector1[0] * vector2[2]
    cross_product[2] = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    return cross_product



def plot_centerline(centerline, scale=0.15):
    """
    Create a 3D plot of the beam's centerline

    Parameters
    ----------
    centerline : array_like
        shape[0] = 3 (dimensions of space)
        spape[1] = n_n (number of nodes)
    scale : float
        plot axis are chosen as:
        -scale/2 <= x <= +scale/2
        -scale/2 <= y <= +scale/2
               0 <= z <= +scale
    """
    xylim = scale / 2
    assert centerline.shape[0] == 3
    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection='3d')
    axis.set_xlim(-xylim, xylim)
    axis.set_ylim(-xylim, xylim)
    axis.set_zlim(0, scale)
    axis.plot(centerline[0, :], centerline[1, :], centerline[2, :])
