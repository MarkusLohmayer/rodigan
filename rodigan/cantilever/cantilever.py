"""
FEM solver for a static cantilever
"""

import numpy as np

from ..common.solver import Solver
from ..common.result import Result

from .newton import newton_rhapson


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
        result_tuple = newton_rhapson(self.number_of_nodes, self.geometry.length,
                                      self.material.elasticity_tensor,
                                      self.boundary_condition,
                                      self.load_control_parameters[0],
                                      self.load_control_parameters[1],
                                      self.maximum_iterations_per_loadstep
                                     )

        # store the results in a container class instance
        self.result = Result(result_tuple)
