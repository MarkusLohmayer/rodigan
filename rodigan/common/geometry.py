"""
Class that stores geometric properties of a special Cosserat rod.
"""

import numpy as np


class Geometry:
    """
    This class holds the geometry of the rod.
    """
    def __init__(self, length, radius):
        """
        Create a new geometry by setting the length and radius of the rod.
        """
        self.length = length
        self.radius = radius


    @property
    def length(self):
        """Length of the rod."""
        return self.__length


    @length.setter
    def length(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError('The length must be given as a positive number!')
        if value <= 0:
            raise ValueError('The length must be positive!')
        # pylint: disable=W0201
        self.__length = float(value)


    @property
    def radius(self):
        """The rod's cross-section radius."""
        return self.__radius


    @radius.setter
    def radius(self, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError('The radius must be given as a positive number!')
        if value <= 0:
            raise ValueError('The radius must be positive!')
        if value >= self.length / 10:
            raise ValueError('The radius is too big. Rods are by definition slender.')
        # pylint: disable=W0201
        self.__radius = float(value)


    @property
    def cross_section_area(self):
        """The rod's cross-section area calculated based on the given radius."""
        return np.pi * self.radius**2


    @property
    def second_moment_of_area(self):
        """The rod's cross-section second moment of area calculated based on the given radius."""
        return np.pi * self.radius**4 / 4
