"""
Class that stores simulation results (produced by a Solver subclass).
"""

import visualization


class Result:
    """
    A container for the results of a simulation run.
    """
    def __init__(self, results_tuple):
        self.__centerline = results_tuple[0]
        self.__load_step_iterations = results_tuple[1]
        self.__load_steps = results_tuple[2]
        self.__residuals_norm_evolution = results_tuple[3]
        self.__increments_norm_evolution = results_tuple[4]



    @property
    def centerline(self):
        """The computed centerline of the last simulation run."""
        return self.__centerline



    def plot_centerline(self):
        """Creates a 3D plot of the beam's centerline."""
        if self.centerline is not None:
            visualization.plot_centerline(self.centerline)
        else:
            print("No centerline was found.")



    @property
    def load_step_iterations(self):
        """
        load_step_iterations : list of integers
            List of iteration counts in every load step.
        """
        return self.__load_step_iterations



    def plot_load_step_iterations(self):
        """Creates a plots about how many iterations each load step took."""
        if self.load_step_iterations is not None:
            visualization.plot_load_step_iterations(self.load_step_iterations)
        else:
            print("No simulation results were found.")



    @property
    def load_steps(self):
        """
        load_steps : list of ndarrays
            List of the applied boundary conditions in every load step.
        """
        return self.__load_steps



    @property
    def residuals_norm_evolution(self):
        """
        residuals_norm_evolution : list of lists of floats
            Every sublist corresponds to one load step and contains the norms
            of the residuals vector in every iteration.
        """
        return self.__residuals_norm_evolution



    @property
    def increments_norm_evolution(self):
        """
        increments_norm_evolution : list of lists of floats
            Every sublist corresponds to one load step and contains the norms
            of the increments vector in every iteration.
        """
        return self.__increments_norm_evolution



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
        if self.load_step_iterations is not None and \
           self.residuals_norm_evolution is not None and \
           self.increments_norm_evolution is not None:
            if -1 <= load_step < len(self.load_step_iterations):
                visualization.plot_norms_in_loadstep(self.residuals_norm_evolution,
                                                     self.increments_norm_evolution,
                                                     load_step)
            else:
                print("The given load step does not exist")
        else:
            print("No simulation results were found.")
