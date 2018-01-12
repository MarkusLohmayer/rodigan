"""
Code for assembly of the residuals vector and the Jacobian matrix.
"""

import numpy as np

import numba
from numba.types import float64, int64, Tuple

from ..common.functions import auxiliary
from ..common.functions import rotations

VEC = float64[:]
MAT = float64[:, :]


# pylint: disable=R0913
# pylint: disable=R0914
@numba.jit(Tuple((VEC, MAT))
           (int64, VEC, MAT, MAT, MAT, VEC, MAT), nopython=True, cache=True)
def assemble_residuals_and_jacobian(number_of_nodes, element_lengths, elasticity_tensor,
                                    centerline, rotation,
                                    increments, second_strain_invariant):
    """
    Assembles the residuals vector and the Jacobian matrix.
    """
    # start with a blank residuals vector
    residuals = np.zeros((6*number_of_nodes), dtype=float64)

    # start with a blank Jacobian matrix
    jacobian = np.zeros((6*number_of_nodes, 6*number_of_nodes), dtype=float64)

    # add contributions of each element
    for i in range(1, number_of_nodes):

        # length of the element
        element_length = element_lengths[i-1]

        # approximate rotation matrix at the midpoint of the element
        rotation_matrix = rotations.interpolate_euler(rotation[:, i-1], rotation[:, i])


        # compute first invariant strain measure in the element ...

        # compute translational displacement tangent vector
        centerline_tangent = (centerline[:, i] - centerline[:, i-1]) / element_length

        # first invariant strain measure
        first_strain_invariant = np.dot(rotation_matrix.T, centerline_tangent)
        # no axial strain <=> lambda_3 = 1
        first_strain_invariant[2] -= 1


        # compute second invariant strain measure in the element using Simo's formula ...

        # incremental rotation tangent vector
        incremental_euler_tangent = \
        (increments[6*i+3:6*i+6] - increments[6*(i-1)+3:6*(i-1)+6]) / element_length

        # incremental Euler vector at midpoint of element
        mid_incremental_euler = (increments[6*i+3:6*i+6] + increments[6*(i-1)+3:6*(i-1)+6]) / 2
        mid_incremental_euler_norm = np.linalg.norm(mid_incremental_euler)

        # compute beta
        if mid_incremental_euler_norm < 1e-6:
            # use asymptotic approximation of Simo's formula to save computational cost
            beta = incremental_euler_tangent + \
                   0.5*auxiliary.cross(mid_incremental_euler, incremental_euler_tangent)
        else:
            fooo = np.sin(mid_incremental_euler_norm) / mid_incremental_euler_norm
            delu = mid_incremental_euler / mid_incremental_euler_norm
            beta = fooo*incremental_euler_tangent + \
                   (1-fooo) * np.dot(delu.T, incremental_euler_tangent) * delu + \
                   2*(np.sin(0.5*mid_incremental_euler_norm) / mid_incremental_euler_norm)**2 \
                   * auxiliary.cross(mid_incremental_euler, incremental_euler_tangent)

        # updating the second strain invariant
        second_strain_invariant[:, i-1] += np.dot(rotation_matrix.T, beta)

        #-----------------

        # compute internal reactions in inertial frame of the element ...
        strain_invariants = np.hstack((first_strain_invariant, second_strain_invariant[:, i-1]))
        forces = np.dot(rotation_matrix, np.dot(elasticity_tensor[0:3, :], strain_invariants))
        moments = np.dot(rotation_matrix, np.dot(elasticity_tensor[3:6, :], strain_invariants))

        #-----------------

        # add contriubutions of the element to residual vector ...

        # contributions from internal forces and moments
        crossphin = 0.5 * element_length * auxiliary.cross(centerline_tangent, forces)
        residuals[6*(i-1):6*i] += np.hstack((-forces, -crossphin - moments))
        residuals[6*i:6*(i+1)] += np.hstack((+forces, -crossphin + moments))


        # add contributions of the element to Jacobian matrix ...

        # symmetrize (because of roundoff error ?)
        C11 = np.dot(np.dot(rotation_matrix, elasticity_tensor[0:3, 0:3]), rotation_matrix.T)
        C11 = (C11 + C11.T) / 2
        C12 = np.dot(np.dot(rotation_matrix, elasticity_tensor[0:3, 3:6]), rotation_matrix.T)
        C21 = C12.T
        C22 = np.dot(np.dot(rotation_matrix, elasticity_tensor[3:6, 3:6]), rotation_matrix.T)
        C22 = (C22 + C22.T) / 2

        centerline_tangent_cross = auxiliary.skew_matrix_from_vector(centerline_tangent)
        forces_cross = auxiliary.skew_matrix_from_vector(forces)
        moments_cross = auxiliary.skew_matrix_from_vector(moments)

        # material tangent stiffness (symmetric part)
        jacobian[6*(i-1):6*i, 6*(i-1):6*i] += np.vstack((np.hstack((+C11 / element_length,
                                                                    -0.5*np.dot(C11, centerline_tangent_cross) + C12 / element_length)),
                                                         np.hstack((-0.5*np.dot(centerline_tangent_cross.T, C11) + C21 / element_length,
                                                                    np.dot(np.dot(centerline_tangent_cross.T, C11), centerline_tangent_cross)*(element_length / 3) - 0.5*np.dot(centerline_tangent_cross.T, C12) + np.dot(C21, centerline_tangent_cross) + C22 / element_length))))

        jacobian[6*i:6*(i+1), 6*i:6*(i+1)] += np.vstack((np.hstack((+C11 / element_length,
                                                                    +0.5*np.dot(C11, centerline_tangent_cross) + C12 / element_length)),
                                                         np.hstack((+0.5*np.dot(centerline_tangent_cross.T, C11) + C21 / element_length,
                                                                    np.dot(np.dot(centerline_tangent_cross.T, C11), centerline_tangent_cross)*(element_length / 3) + 0.5*np.dot(centerline_tangent_cross.T, C12) + np.dot(C21, centerline_tangent_cross) + C22 / element_length))))

        jacobian[6*i:6*(i+1), 6*(i-1):6*i] += np.vstack((np.hstack((-C11 / element_length,
                                                                    +0.5*np.dot(C11, centerline_tangent_cross) - C12 / element_length)),
                                                         np.hstack((-0.5*np.dot(centerline_tangent_cross.T, C11) - C21 / element_length,
                                                                    np.dot(np.dot(centerline_tangent_cross.T, C11), centerline_tangent_cross)*(element_length / 6) - 0.5*np.dot(centerline_tangent_cross.T, C12) - np.dot(C21, centerline_tangent_cross) - C22 / element_length))))

        jacobian[6*(i-1):6*i, 6*i:6*(i+1)] += np.vstack((np.hstack((-C11 / element_length,
                                                                    -0.5*np.dot(C11, centerline_tangent_cross) - C12 / element_length)),
                                                         np.hstack((+0.5*np.dot(centerline_tangent_cross.T, C11) - C21 / element_length,
                                                                    np.dot(np.dot(centerline_tangent_cross.T, C11), centerline_tangent_cross)*(element_length / 6) + 0.5*np.dot(centerline_tangent_cross.T, C12) - np.dot(C21, centerline_tangent_cross) - C22 / element_length))))

        # geometric tangent stiffness (non-symmetric)
        jacobian[6*(i-1):6*i, 6*(i-1):6*i] += np.vstack((np.hstack((np.zeros((3, 3)), +0.5*forces_cross)),
                                                         np.hstack((-0.5*forces_cross, +0.5*moments_cross - np.dot(centerline_tangent_cross.T, forces_cross)*(element_length / 3)))))

        jacobian[6*i:6*(i+1), 6*i:6*(i+1)] += np.vstack((np.hstack((np.zeros((3, 3)), -0.5*forces_cross)),
                                                         np.hstack((+0.5*forces_cross, -0.5*moments_cross - np.dot(centerline_tangent_cross.T, forces_cross)*(element_length / 3)))))

        jacobian[6*i:6*(i+1), 6*(i-1):6*i] += np.vstack((np.hstack((np.zeros((3, 3)), -0.5*forces_cross)),
                                                         np.hstack((-0.5*forces_cross, -0.5*moments_cross - np.dot(centerline_tangent_cross.T, forces_cross)*(element_length / 6)))))

        jacobian[6*(i-1):6*i, 6*i:6*(i+1)] += np.vstack((np.hstack((np.zeros((3, 3)), +0.5*forces_cross)),
                                                         np.hstack((+0.5*forces_cross, +0.5*moments_cross - np.dot(centerline_tangent_cross.T, forces_cross)*(element_length / 6)))))

        # tangent due to distributive load

        # tangent due to boundary loads

    return residuals, jacobian
