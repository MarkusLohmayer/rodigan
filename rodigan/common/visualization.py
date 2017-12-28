""" Routines for visualizing results """

#import numpy as np

import itertools

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=W0611



def plot_centerline(centerline, scale=0.15):
    """
    Creates a 3D plot of the beam's centerline.

    Parameters
    ----------
    centerline : array_like
        shape[0] = 3 (dimensions of space)
        spape[1] = n_n (number of nodes)
    scale : float
        plot axis are chosen as:
        -scale/2 <= x <= +scale/2
        -scale/2 <= y <= +scale/2
               0 <= z <= +scale
    """
    assert centerline.shape[0] == 3

    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection='3d')

    # set range of axes
    xylim = scale / 2
    axis.set_xlim(-xylim, xylim)
    axis.set_ylim(-xylim, xylim)
    axis.set_zlim(0, scale)

    # label axes
    axis.set_xlabel(r'$e_1$')
    axis.set_ylabel(r'$e_2$')
    axis.set_zlabel(r'$e_3$')

    # plot
    axis.plot(centerline[0, :], centerline[1, :], centerline[2, :])



def plot_norms_in_loadstep(residuals_norm_evolution, increments_norm_evolution, load_step=0):
    """
    Plots the evolution of residuals_norm and increments_norm in a given loadstep.

    Parameters
    ----------
    residuals_norm_evolution : list of lists of floats
    increments_norm_evolution : list of lists of floats
    load_step : int
        If -1 is chosen then all load steps are concatenated in one plot.
    """
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111)
    secondary_axis = axis.twinx()

    # label axes
    axis.set_xlabel('iteration')
    axis.set_ylabel('norm of residuals')
    secondary_axis.set_ylabel('norm of increments')

    # only integer ticks for iterations
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    # define colors
    color1 = plt.cm.viridis(0.5) # pylint: disable=E1101
    color2 = plt.cm.viridis(0.9) # pylint: disable=E1101

    # plot
    if load_step == -1:
        # merge all sublists into one list
        all_residual_norms = list(itertools.chain.from_iterable(residuals_norm_evolution))
        all_increment_norms = list(itertools.chain.from_iterable(increments_norm_evolution))

        resiudals_plot = axis.plot(all_residual_norms,
                                   color=color1, label="residuals")
        increments_plot = secondary_axis.plot(all_increment_norms,
                                              color=color2, label="increments")
    else:
        resiudals_plot = axis.plot(residuals_norm_evolution[load_step],
                                   color=color1, label="residuals")
        increments_plot = secondary_axis.plot(increments_norm_evolution[load_step],
                                              color=color2, label="increments")

    # add a legend
    #legend_handles = [resiudals_plot[0], increments_plot[0]]
    #axis.legend(handles=legend_handles, loc='best')

    # color axis labels like corresponding curves
    axis.yaxis.label.set_color(resiudals_plot[0].get_color())
    secondary_axis.yaxis.label.set_color(increments_plot[0].get_color())



def plot_load_step_iterations(load_step_iterations):
    """
    Plots the number of iterations each loadstep took.

    Parameters
    ----------
    load_step_iterations : list of ints
    """
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111)

    # only integer ticks
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    axis.bar(range(len(load_step_iterations)), load_step_iterations)
