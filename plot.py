import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

from utils import Re, inv_Re

def format_plots(dpi=150, use_tex=True, font_type='serif', font_family='Computer Modern', font_size=15):
    rc('figure', dpi=dpi)
    rc('font', **{'family': font_type, font_type: [font_family], 'size': font_size})
    rc('text', usetex=use_tex)
    return


def plot_field(field, h_axis, v_axis, x_lims, y_lims, nodes=20, size=5, t=0., labels=('x', 'y'), title='Field Lines'):
    '''
    Creates a plot of the integral curves of a field along a plane passing through the origin.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    h_axis : float[3]
        The vector coinciding with the horizontal axis.

    v_axis : float[3]
        The vector coinciding with the vertical axis.        

    x_lims : float[2]:
        A 2 element list consisting of the horizontal axis limits. The lower value is listed first.

    y_lims : float[2]
        A 2 element list consisting of the vertical axis limits. The lower value is listed first.

    nodes : int
        The number of samples to be taken along each axis.

    size : float, optional
        The scaling factor for the figure dimensions. Defaults to 10.

    t : float, optional
        The time at which to evaluate the field. Defaults to 0.

    labels : (string, string), optional
        The horizontal and vertical labels of the plot. Defaults to ('x', 'y').

    title : string, optional
        The title of the plot. Defaults to 'Field Lines'.

    Returns
    -------
    None
    '''

    aspect_ratio = (y_lims[1] - y_lims[0]) / (x_lims[1] - x_lims[0])

    x = np.linspace(x_lims[0], x_lims[1], nodes)
    y = np.linspace(y_lims[0], y_lims[1], nodes)

    h_axis = h_axis / np.linalg.norm(h_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(size, size * aspect_ratio))

    for i in range(nodes):
        for j in range(nodes):
            vec_l = field(np.asarray((h_axis * X[i][j] + v_axis * Y[i][j]) * Re), t)

            U[i][j] = np.dot(vec_l, h_axis)
            V[i][j] = np.dot(vec_l, v_axis)

    
    ax.set_title(title)
    
    ax.set_xlabel(labels[0] + '($R_E$)')
    ax.set_ylabel(labels[1] + '($R_E$)')

    color = 2 * np.log(np.hypot(U, V))
    ax.streamplot(X, Y, U, V, color=color, linewidth=1, cmap=plt.cm.plasma, density=2, arrowstyle='wedge', arrowsize=1.)

    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])

    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_distribution(history, time_ind, size=5, labels=('Quantity', 'Density (AU)'), title='Distribution', log=False):
    fig = plt.figure(figsize=(size, size))

    plt.hist(history[:, time_ind], bins='fd', density=True)
    plt.autoscale(enable=True, tight=True)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if log == True:
        plt.yscale('log', nonposy='clip')

    plt.title(title)

    plt.grid()
    plt.tight_layout()
    plt.show()

    return


def plot_evolution(history, time, size=5, labels=('Time (s), Quantity, Strength (AU)'), title='Evolution', log=True):
    fig = plt.figure(figsize=(size, size))

    plt.hist(history[:, time_ind], bins='fd', density=True)
    plt.autoscale(enable=True, tight=True)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if log == True:
        plt.yscale('log', nonposy='clip')

    plt.title(title)

    plt.grid()
    plt.tight_layout()
    plt.show()

    return
