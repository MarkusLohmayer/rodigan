"""
abstract base class for a generic special Cosserat rod FEM solver
"""

from abc import ABCMeta, abstractmethod

from .material import Material
from .geometry import Geometry
from .result import Result



class Solver(metaclass=ABCMeta):
    """
    This class holds parameters of a FEM discretization,
    configuration parameters of the FEM solver,
    calls the FEM solver code in `run_simulation` (abstractmethod),
    and stores the results in a `Result` object, referenced by `self.result`.
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

        # result from last call to `run_simulation`
        self.__result = None



    @property
    def geometry(self):
        """The reference to an instance of Geometry, where the rod's geometry is defined."""
        return self.__geometry


    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):
            raise TypeError('Please reference an instance of the class Geometry!')
        self.__geometry = value # pylint: disable=W0201



    @property
    def material(self):
        """The reference to an instance of Material, where the rod's geometry is defined."""
        return self.__material


    @material.setter
    def material(self, value):
        if not isinstance(value, Material):
            raise TypeError('Please reference an instance of the class Material!')
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
    @abstractmethod
    def boundary_condition(self):
        """problem specific boundary conditions"""
        pass


    @boundary_condition.setter
    @abstractmethod
    def boundary_condition(self, value):
        pass



    @abstractmethod
    def run_simulation(self):
        """run the problem specific FEM solver code
        and store results as a `Result` object in self.result"""



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
        If the parameter is set to -1 then no threshold is imposed."""
        return self.__maximum_iterations_per_loadstep


    @maximum_iterations_per_loadstep.setter
    def maximum_iterations_per_loadstep(self, value):
        if isinstance(value, int) and value >= -1:
            self.__maximum_iterations_per_loadstep = value
        else:
            raise ValueError('Expected an integer! (value >= -1)')



    @property
    def result(self):
        """
        A reference to an instance of the Result class.
        """
        return self.__result


    @result.setter
    def result(self, value):
        if not isinstance(value, Result):
            raise TypeError('Please reference an instance of the class Result!')
        self.__result = value # pylint: disable=W0201
