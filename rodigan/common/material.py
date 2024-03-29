"""
Class that stores material properties of a special Cosserat rod.
"""

import numpy as np

from .geometry import Geometry


class Material:
    """
    This class holds the material properties of the rod.
    """
    def __init__(self, geometry, elastic_modulus, shear_modulus):
        """
        Create a new material by referencing a geometry and setting the
        elastic modulus and shear modulus of the rod.
        """
        self.geometry = geometry
        self.elastic_modulus = elastic_modulus
        self.shear_modulus = shear_modulus


    @property
    def geometry(self):
        """The reference to an instance of Geometry, where the rod's geometry is defined."""
        return self.__geometry


    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):
            raise TypeError('Please reference an instance of the class geometry.Geometry!')
        # pylint: disable=W0201
        self.__geometry = value


    @property
    def elastic_modulus(self):
        """The elastic modulus of the rod's material."""
        return self.__elastic_modulus


    @elastic_modulus.setter
    def elastic_modulus(self, value):
        if value <= 0:
            raise ValueError('The elastic modulus must be a positive floating point number!')
        # pylint: disable=W0201
        self.__elastic_modulus = float(value)


    @property
    def shear_modulus(self):
        """The shear modulus of the rod's material."""
        return self.__shear_modulus


    @shear_modulus.setter
    def shear_modulus(self, value):
        if value <= 0:
            raise ValueError('The shear modulus must be a positive floating point number!')
        # pylint: disable=W0201
        self.__shear_modulus = float(value)


    @property
    def bending_stiffness(self):
        """The bending stiffness of the rod (also called parameter A)."""
        return self.elastic_modulus * self.geometry.second_moment_of_area


    @property
    def twisting_stiffness(self):
        """The twisting stiffness of the rod (also called parameter B)."""
        # ?? why the factor 2
        return 2 * self.shear_modulus * self.geometry.second_moment_of_area


    @property
    def shearing_stiffness(self):
        """The shearing stiffness of the rod (also called parameter C)."""
        return self.shear_modulus * self.geometry.cross_section_area


    @property
    def extensional_stiffness(self):
        """The extensional stiffness of the rod (also called parameter D)."""
        return self.elastic_modulus * self.geometry.cross_section_area


    @property
    def extensional_twisting_stiffness(self):
        """The coupling between axial strain and twisting of the rod (also called parameter E)."""
        return 0 #np.sqrt(self.extensional_stiffness * self.twisting_stiffness)


    @property
    def bending_shearing_stiffness(self):
        """The coupling between bending and shearing of the rod (also called parameter F)."""
        return 0 #np.sqrt(self.bending_stiffness * self.shearing_stiffness)


    @property
    def elasticity_tensor(self):
        """The elasticity tensor of the rod."""
        matrix = np.diag([self.shearing_stiffness, self.shearing_stiffness,
                          self.extensional_stiffness,
                          self.bending_stiffness, self.bending_stiffness,
                          self.twisting_stiffness])
        # add coupling terms
        matrix[0, 3] = matrix[3, 0] = \
        matrix[1, 4] = matrix[4, 1] = self.bending_shearing_stiffness / 2
        matrix[2, 5] = matrix[5, 2] = self.extensional_twisting_stiffness / 2
        return matrix
