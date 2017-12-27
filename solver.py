"""
FEM solver for static (special Cosserat) rod
"""

import numba
import numpy as np

import auxiliary
import constitutive
import kinematics
import visualization


class StaticSolver:
    """
    This class holds the general parameters of the FEM discretization of the problem
    and should serve only as a base class.
    """
    def __init__(self, geometry, material, number_of_elements):
        """
        Create a new FEM discretization by referencing a geometry, a material and
        specifying the number of elements.
        """
        # parameters that must be set by the constructor
        self.geometry = geometry
        self.material = material
        self.number_of_elements = number_of_elements

        # further parameters that can be manipulated after initialization
        self.__load_control_parameters = 1e-4, 1e-4
        self.__maximum_iterations_per_loadstep = 100

        # results from last call to `run_simulation`
        self._centerline = None
        self._load_step_iterations = None
        self._load_steps = None
        self._residuals_norm_evolution = None
        self._increments_norm_evolution = None



    @property
    def geometry(self):
        """The reference to an instance of Geometry, where the rod's geometry is defined."""
        return self.__geometry


    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, kinematics.Geometry):
            raise TypeError('Please reference an instance of the class kinematics.Geometry!')
        self.__geometry = value # pylint: disable=W0201


    @property
    def material(self):
        """The reference to an instance of Material, where the rod's geometry is defined."""
        return self.__material


    @material.setter
    def material(self, value):
        if not isinstance(value, constitutive.Material):
            raise TypeError('Please reference an instance of the class constitutive.Material!')
        self.__material = value # pylint: disable=W0201


    @property
    def number_of_elements(self):
        """The number or elements of the FEM discretization."""
        return self.__number_of_elements


    @number_of_elements.setter
    def number_of_elements(self, value):
        if not isinstance(value, int) or value <= 2:
            raise ValueError('The number of elements must be a positive integer number')
        self.__number_of_elements = value # pylint: disable=W0201


    @property
    def number_of_nodes(self):
        """The number of nodes of the FEM discretization."""
        return self.number_of_elements + 1


    @property
    def load_control_parameters(self):
        """The thresholds for the norm of the residual vector and the norm of
        the increments vector given as a tuple of two floating point numbers.
        If the norms become smaller than the given thresholds, then the load is increased."""
        return self.__load_control_parameters


    @load_control_parameters.setter
    def load_control_parameters(self, value):
        if isinstance(value, tuple) and len(value) == 2 and \
           isinstance(value[0], float) and isinstance(value[1], float):
            self.__load_control_parameters = value
        else:
            raise ValueError('Expected a tuple of two floating point numbers!')


    @property
    def maximum_iterations_per_loadstep(self):
        """Number of iterations that are allowed within one loadstep.
        If the parameter is set to None then there is no threshold."""
        return self.__maximum_iterations_per_loadstep


    @maximum_iterations_per_loadstep.setter
    def maximum_iterations_per_loadstep(self, value):
        if (isinstance(value, int) and value > 0) or value is None:
            self.__maximum_iterations_per_loadstep = value
        else:
            raise ValueError('Expected a positive integer or None!')


    @property
    def centerline(self):
        """The computed centerline of the last simulation run."""
        return self._centerline


    def plot_centerline(self):
        """Creates a 3D plot of the beam's centerline."""
        if self.centerline is not None:
            visualization.plot_centerline(self.centerline)
        else:
            print("No simulation results were found.")


    @property
    def load_step_iterations(self):
        """
        load_step_iterations : list of integers
            List of iteration counts in every load step.
        """
        return self._load_step_iterations


    def plot_load_step_iterations(self):
        """Creates a plots about how many iterations each load step took."""
        if self._load_step_iterations is not None:
            visualization.plot_load_step_iterations(self._load_step_iterations)
        else:
            print("No simulation results were found.")


    @property
    def load_steps(self):
        """
        load_steps : list of ndarrays
            List of the applied boundary conditions in every load step.
        """
        return self._load_steps


    @property
    def residuals_norm_evolution(self):
        """
        residuals_norm_evolution : list of lists of floats
            Every sublist corresponds to one load step and contains the norms
            of the residuals vector in every iteration.
        """
        return self._residuals_norm_evolution


    @property
    def increments_norm_evolution(self):
        """
        increments_norm_evolution : list of lists of floats
            Every sublist corresponds to one load step and contains the norms
            of the increments vector in every iteration.
        """
        return self._increments_norm_evolution


    def plot_norms_in_loadstep(self, load_step=-1):
        """
        Plots the evolution of residuals_norm and increments_norm in a given loadstep.

        Parameters
        ----------
        load_step : int
            Number of load step.
            If -1 is chosen then all load steps are concatenated in one plot.
        """
        if not isinstance(load_step, int):
            raise TypeError('load_step must be an integer!')
        if self._load_step_iterations is not None and \
           self._residuals_norm_evolution is not None and \
           self._increments_norm_evolution is not None:
            if -1 <= load_step < len(self.load_step_iterations):
                visualization.plot_norms_in_loadstep(self._residuals_norm_evolution,
                                                     self._increments_norm_evolution,
                                                     load_step)
            else:
                print("The given load step does not exist")
        else:
            print("No simulation results were found.")



class Cantilever(StaticSolver):
    """
    This class provides the functionality for solving a static cantilever rod problem.
    The leftmost node is clamped (zero displacement, zero rotation).
    The rightmost node is loaded by forces and/or moments.
    """
    def __init__(self, geometry, material, number_of_elements=100, boundary_condition=None):
        super().__init__(geometry, material, number_of_elements)
        self.__boundary_condition = np.zeros((6), dtype=float)
        if boundary_condition is not None:
            self.boundary_condition = boundary_condition


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
        Runs the simulation and stores the results in the attributes centerline
        and convergence_metrics.
        """
        if self.boundary_condition is None:
            raise RuntimeError('A boundary condition must first be specified!')
        if self.material.geometry is not self.geometry:
            raise RuntimeError('The material is referencing a different geometry!')
        results = newton_rhapson(self.number_of_nodes, self.geometry.length,
                                 self.material.elasticity_tensor,
                                 self.boundary_condition,
                                 self.load_control_parameters[0],
                                 self.load_control_parameters[1],
                                 self.maximum_iterations_per_loadstep
                                )
        self._centerline = results[0]
        self._load_step_iterations = results[1]
        self._load_steps = results[2]
        self._residuals_norm_evolution = results[3]
        self._increments_norm_evolution = results[4]



#@numba.jit(numba.void(numba.int64, numba.float64[:], numba.float64[:, :], numba.float64[:, :],
#numba.float64[:], numba.float64[:, :], numba.float64[:], numba.float64[:, :],
# numba.float64[:, :]), nopython=True, cache=True)
def assembly(i, element_length, elasticity_tensor, centerline, rotation,
             increments, second_strain_invariant, residuals, jacobian):
    """
    Updates configuration in element i and
    adds the contributions of element i to the residuals vector and the Jacobian matrix.
    """
    # update configuration of element ...

    # update centerline at right node of element
    centerline[:, i] += increments[6*i:6*i+3]

    # update rotation at right node of element
    rotation[:, i] = kinematics.update_euler(rotation[:, i], increments[6*i+3:6*i+6])

    if np.linalg.norm(rotation[:, i]) > np.pi:
        print('current rotation larger than pi')

    if np.linalg.norm(increments[6*i+3:6*i+6]) > np.pi:
        print('current incremental rotation larger than pi')

    #-----------------

    # compute rotation matrix at midpoint of element ...

    # note that this interpolation is an approximation
#         mid_euler = (rotation[:, i-1] + rotation[:, i]) / 2
#         if np.linalg.norm(rotation[:, i] - rotation[:, i-1]) > np.pi:
#             # required because extracted axial vector may differ by 2*pi
#             print("a strange thing happened")

#             # this is a hack!
#             mid_euler = rotation[:, i-1]
#         R = rotations.matrix_from_euler(mid_euler)
    R = kinematics.interpolate_euler(rotation[:, i-1], rotation[:, i])

    #-----------------

    # compute first invariant strain measure in the element ...

    # compute translational displacement tangent vector
    centerline_tangent = (centerline[:, i] - centerline[:, i-1]) / element_length

    # first invariant strain measure
    first_strain_invariant = np.dot(R.T, centerline_tangent)
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
    second_strain_invariant[:, i-1] += np.dot(R.T, beta)

    #-----------------

    # compute internal reactions in inertial frame of the element ...

    forces = np.dot(R, np.array([elasticity_tensor[0, 0]*first_strain_invariant[0],
                                 elasticity_tensor[1, 1]*first_strain_invariant[1],
                                 elasticity_tensor[2, 2]*first_strain_invariant[2]]))

    moments = np.dot(R, np.array([elasticity_tensor[3, 3]*second_strain_invariant[0, i-1],
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
    C11 = np.dot(np.dot(R, elasticity_tensor[0:3, 0:3]), R.T)
    C11 = (C11 + C11.T) / 2
    C12 = np.dot(np.dot(R, elasticity_tensor[0:3, 3:6]), R.T)
    C21 = C12.T
    C22 = np.dot(np.dot(R, elasticity_tensor[3:6, 3:6]), R.T)
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



#@numba.jit(numba.float64[:, :](numba.int64, numba.int64, numba.float64[:], numba.float64,
#numba.float64, numba.float64, numba.float64[:, :]), nopython=True, cache=True)
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

        # start with a blank residuals vector
        residuals = np.zeros((6*number_of_nodes), dtype=float) #numba.float64

        # start with a blank Jacobian matrix
        jacobian = np.zeros((6*number_of_nodes, 6*number_of_nodes), dtype=float) #numba.float64

        # assemble residuals vector and Jacobian matrix
        for i in range(1, number_of_nodes):
            element_length = element_lengths[i-1]
            assembly(i, element_length, elasticity_tensor, centerline, rotation,
                     increments, second_strain_invariant, residuals, jacobian)

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
        if load_step_iterations[-1] > maximum_iterations_per_loadstep:
            raise RuntimeError('Stopped execution after reaching maximum number'
                               ' of iterations in this load step!')

    return centerline, load_step_iterations, load_steps, \
           residuals_norm_evolution, increments_norm_evolution
