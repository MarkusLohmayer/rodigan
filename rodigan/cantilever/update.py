"""
Code for updating the rod's configuration.
"""

import numpy as np

import numba
from numba.types import float64, int64

from ..common.functions import rotations

VEC = float64[:]
MAT = float64[:, :]

@numba.jit((int64, MAT, MAT, VEC), nopython=True, cache=True)
def update_configuration(number_of_nodes, centerline, rotation, increments):
    """Updates configuration."""
    for i in range(1, number_of_nodes):
        # update centerline at right node of element
        centerline[:, i] += increments[6*i:6*i+3]

        # update rotation at right node of element
        rotation[:, i] = rotations.compose_euler(rotation[:, i], increments[6*i+3:6*i+6])

        # watch out for strange things that might happen ...
        if np.linalg.norm(rotation[:, i]) > np.pi:
            print('current rotation larger than pi')

        if np.linalg.norm(increments[6*i+3:6*i+6]) > np.pi:
            print('current incremental rotation larger than pi')
