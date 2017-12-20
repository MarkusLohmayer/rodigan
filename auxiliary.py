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
