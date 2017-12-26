""" Functions for visualizing results """

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D # pylint: disable=W0611

def plot_centerline(centerline, scale=0.15):
    """
    Create a 3D plot of the beam's centerline

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
