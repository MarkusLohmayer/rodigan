"""
Code for Newton-Rhapson scheme.
"""

import numpy as np
from numpy import float64 # pylint: disable=E0611

#import numba
# from numba.types import float64, int64, Tuple, List

from .update import update_configuration
from .assembly import assemble_residuals_and_jacobian

# VEC = float64[:]
# MAT = float64[:, :]

# pylint: disable=R0913
# pylint: disable=R0914
# @numba.jit(Tuple((MAT,
#                   List(int64, reflected=True),
#                   List(VEC, reflected=True),
#                   List(List(float64, reflected=True), reflected=True),
#                   List(List(float64, reflected=True), reflected=True)))
#            (int64, float64, MAT, VEC, float64, float64, int64),
#            nopython=True, cache=True)
# # numba: reflected list(reflected list(float64))
# # unsupported nested memory-managed object
def newton_rhapson(number_of_nodes, length, elasticity_tensor, boundary_condition,
                   load_control_residuals, load_control_increments,
                   maximum_iterations_per_loadstep):
    """
    This function contains initialization of required variables and the outer
    loop of a Newton-Rhapson scheme for solving the nonlinear FEM problem.
    """
    number_of_elements = number_of_nodes - 1

    # length of elements
    element_lengths = np.ones((number_of_elements)) * (length / number_of_elements)

    # centerline displacement at each node
    centerline = np.zeros((3, number_of_nodes), dtype=float64)
    for i in range(number_of_elements):
        centerline[2, i+1] = centerline[2, i] + element_lengths[i]

    # rotation of crosssection at each node,
    # axis-angle representation: 1 Euler vector per node
    rotation = np.zeros((3, number_of_nodes), dtype=float64)

    # displacement increment vector
    # 3 centerline displacement variables + 3 cross-section rotation variables per node
    increments = np.zeros((6 * number_of_nodes), dtype=float64)

    # must be stored persistently, so we can use Simo's update formula
    second_strain_invariant = np.zeros((3, number_of_elements), dtype=float64)

    # boundary_condition in each load step
    load_steps = []

    # counting iterations in each load step
    load_step_iterations = []

    # for keeping track of convergence
    # list of lists: one list per load step
    residuals_norm_evolution = []
    increments_norm_evolution = []

    # we start simulation without any load and then increase the load gradually
    # -> load-controlled Newton-Rhapson
    current_load = np.zeros((6), dtype=float64)

    # flag that marks if load step has converged
    converged = True

    # Newton-Rhapson iterations
    while np.max(np.abs(boundary_condition - current_load)) > 0.1 or not converged:

        # when converged begin next load step
        if converged:
            # increase loading
            # (last load step serves as initial condition for new load step)
            current_load += 0.1 * np.sign(boundary_condition - current_load)

            # bookkeping stuff (on a per-load-step basis)
            # iteration count in new load step
            load_step_iterations.append(0)
            # boundary condition in new load step
            load_steps.append(np.copy(current_load))
            # evolution of convergence indicators in new load step
            residuals_norm_evolution.append([])
            increments_norm_evolution.append([])


        # count iterations for current load step
        load_step_iterations[-1] += 1

        # stop execution after maximum iterations in one load step
        if maximum_iterations_per_loadstep > 0:
            if load_step_iterations[-1] > maximum_iterations_per_loadstep:
                raise RuntimeError('Stopped execution after reaching maximum number'
                                   ' of iterations in this load step!')

        # assemble residuals vector and Jacobian matrix
        residuals, jacobian = assemble_residuals_and_jacobian(number_of_nodes,
                                                              element_lengths, elasticity_tensor,
                                                              centerline, rotation,
                                                              increments, second_strain_invariant)

        # apply Neumann boundary conditions
        residuals[-6:] -= current_load

        # solve the linearized problem
        increments[6:] = np.linalg.solve(-jacobian[6:, 6:], residuals[6:])

        # compute norm of residuals vector
        residuals_norm = np.linalg.norm(residuals[6:])
        residuals_norm_evolution[-1].append(residuals_norm)

        # compute norm of increments vector
        increments_norm = np.linalg.norm(increments[6:])
        increments_norm_evolution[-1].append(increments_norm)

        # don't allow large increments
        if increments_norm > 1:
            print('normalized increments!')
            increments[6:] = increments[6:] / increments_norm

        # update the configuration for the next iteration
        update_configuration(number_of_nodes, centerline, rotation, increments)

        # finally, evaluate condition for convergence
        converged = residuals_norm < load_control_residuals and \
                    increments_norm < load_control_increments

    return centerline, load_step_iterations, load_steps, \
           residuals_norm_evolution, increments_norm_evolution
