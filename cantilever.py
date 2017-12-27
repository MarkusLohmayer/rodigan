"""
FEM solver for a static cantilever
"""

import numba
from numba.types import float64
from numba.types import int64

import numpy as np

import auxiliary
import rotations

from solver import Solver
from result import Result


class Cantilever(Solver):
    """
    This class provides the functionality for solving a static cantilever rod problem.
    The leftmost node is clamped (zero displacement, zero rotation).
    The rightmost node is loaded by forces and/or moments.
    """
    def __init__(self, geometry, material, number_of_elements=100, boundary_condition=None):
        super().__init__(geometry, material, number_of_elements)

        # BC: reference to None or to a ndarray of six floats
        self.__boundary_condition = None
        if boundary_condition is not None:
            self.boundary_condition = boundary_condition

        # result from last call to `run_simulation`
        self.__result = None


    @property
    def boundary_condition(self):
        """
        The six loading conditions at the rightmost node.
        First 3 numbers are forces and last 3 numbers are moments.
        """
        return self.__boundary_condition


    @boundary_condition.setter
    def boundary_condition(self, value):
        if isinstance(value, list):
            value = np.array(value, dtype=float)
        if not isinstance(value, np.ndarray) or value.shape != (6,):
            raise ValueError('The boundary condition must be an array of six numbers.')
        if value.dtype != np.float64: # pylint: disable=E1101
            value = value.astype(np.float64) # pylint: disable=E1101
        self.__boundary_condition = value


    def run_simulation(self):
        """
        Runs the simulation.
        Results are stored in the container class instance `self.results`.
        """
        if self.boundary_condition is None:
            raise RuntimeError('A boundary condition must first be specified!')

        if self.material.geometry is not self.geometry:
            raise RuntimeError('The material is referencing a different geometry!')

        # run simulation
        result = newton_rhapson(self.number_of_nodes, self.geometry.length,
                                self.material.elasticity_tensor,
                                self.boundary_condition,
                                self.load_control_parameters[0],
                                self.load_control_parameters[1],
                                self.maximum_iterations_per_loadstep
                               )

        # store the results in a container class instance
        self.__result = Result(result)


    @property
    def result(self):
        """
        A reference to an instance of the Result class.
        """
        return self.__result



@numba.jit((int64, float64[:, :], float64[:, :], float64[:]),
           nopython=True, cache=True)
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



@numba.jit(numba.types.Tuple((float64[:], float64[:, :]))
           (int64, float64[:], float64[:, :], float64[:, :], float64[:, :],
            float64[:], float64[:, :]), nopython=True, cache=True)
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
        incremental_euler_tangent = (increments[6*i+3:6*i+6] - increments[6*(i-1)+3:6*(i-1)+6]) / element_length

        # incremental Euler vector at midpoint of element
        mid_incremental_euler = (increments[6*i+3:6*i+6] + increments[6*(i-1)+3:6*(i-1)+6]) / 2
        norm_of_mid_incremental_euler = np.linalg.norm(mid_incremental_euler)

        # compute beta
        if norm_of_mid_incremental_euler < 1e-6:
            # use asymptotic approximation of Simo's formula to save computational cost
            beta = incremental_euler_tangent + \
                   0.5*auxiliary.cross(mid_incremental_euler, incremental_euler_tangent)
        else:
            x = np.sin(norm_of_mid_incremental_euler) / norm_of_mid_incremental_euler
            delu = mid_incremental_euler / norm_of_mid_incremental_euler
            beta = x*incremental_euler_tangent + \
                   (1-x) * np.dot(delu.T, incremental_euler_tangent) * delu + \
                   2 * (np.sin(0.5*norm_of_mid_incremental_euler) / norm_of_mid_incremental_euler)**2 * auxiliary.cross(mid_incremental_euler, incremental_euler_tangent)

        # updating the second strain invariant
        second_strain_invariant[:, i-1] += np.dot(rotation_matrix.T, beta)

        #-----------------

        # compute internal reactions in inertial frame of the element ...

        forces = np.dot(rotation_matrix,
                        np.array([elasticity_tensor[0, 0]*first_strain_invariant[0],
                                  elasticity_tensor[1, 1]*first_strain_invariant[1],
                                  elasticity_tensor[2, 2]*first_strain_invariant[2]]))

        moments = np.dot(rotation_matrix,
                         np.array([elasticity_tensor[3, 3]*second_strain_invariant[0, i-1],
                                   elasticity_tensor[4, 4]*second_strain_invariant[1, i-1],
                                   elasticity_tensor[5, 5]*second_strain_invariant[2, i-1]]))

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



#@numba.jit(nopython=True, cache=True)
def newton_rhapson(number_of_nodes, length, elasticity_tensor, boundary_condition,
                   load_control_parameter_residuals, load_control_parameter_increments,
                   maximum_iterations_per_loadstep):
    """
    This function contains initialization of required variables and the outer
    loop of a Newton-Rhapson scheme for solving the nonlinear FEM problem.
    """
    number_of_elements = number_of_nodes - 1

    # length of elements
    element_lengths = np.ones((number_of_elements)) * (length / number_of_elements)

    # centerline displacement at each node
    centerline = np.zeros((3, number_of_nodes), dtype=float) #numba.float64
    for i in range(number_of_elements):
        centerline[2, i+1] = centerline[2, i] + element_lengths[i]

    # rotation of crosssection at each node,
    # axis-angle representation: 1 Euler vector per node
    rotation = np.zeros((3, number_of_nodes), dtype=float) #numba.float64

    # displacement increment vector
    # 3 centerline displacement variables + 3 cross-section rotation variables per node
    increments = np.zeros((6 * number_of_nodes), dtype=float) #numba.float64

    # must be stored persistently, so we can use Simo's update formula
    second_strain_invariant = np.zeros((3, number_of_elements), dtype=float) #numba.float64

    # boundary_condition in each load step
    load_steps = []

    # counting iterations in each load step
    load_step_iterations = []

    # for keeping track of convergence
    # list of lists: one list per load step
    residuals_norm = 0
    residuals_norm_evolution = []
    increments_norm = 0
    increments_norm_evolution = []

    # we start simulation without any load and then increase the load gradually
    # -> load-controlled Newton-Rhapson
    target_load = boundary_condition
    current_load = np.zeros((6), dtype=float) #numba.float64

    # Newton-Rhapson iterations
    while np.max(np.abs(target_load - current_load)) > 0.1:
        # check for convergence to possibly begin next load step
        if residuals_norm < load_control_parameter_residuals and \
           increments_norm < load_control_parameter_increments:

            # increase loading ->
            # current load step serves as initial condition for next load step
            load_change = 0.1 * np.sign(target_load - current_load)
            current_load += load_change

            # iteration counts and convergence indicators on a per-load-step basis
            load_step_iterations.append(0)
            load_steps.append(np.copy(current_load))
            residuals_norm_evolution.append([])
            increments_norm_evolution.append([])


        # count iterations for current load step
        load_step_iterations[-1] += 1

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

        # stop execution after maximum iterations in one load step
        if maximum_iterations_per_loadstep > 0:
            if load_step_iterations[-1] > maximum_iterations_per_loadstep:
                raise RuntimeError('Stopped execution after reaching maximum number'
                                   ' of iterations in this load step!')

        # update the configuration for the next iteration
        update_configuration(number_of_nodes, centerline, rotation, increments)

    return centerline, load_step_iterations, load_steps, \
           residuals_norm_evolution, increments_norm_evolution
