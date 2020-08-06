import numpy as np
from matplotlib import pyplot as plt


def plot_field(field, axis, nodes, x_lims, y_lims, size=(10, 10), t=0.):
    '''
    Creates a plot of the integral curves of a field along one of the three axis-aligned planes passing through the origin.

    Parameters
    ----------
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    axis (string): Either 'x', 'y', or 'z.' This is the axis the plane of the graph will be orthogonal to.
    nodes (int): The number of samples to be taken along each axis.
    x_lims (2 list): A 2 element list consisting of the horizontal axis limits. The lower value is listed first.
    y_lims (2 list): A 2 element list consisting of the vertical axis limits. The lower value is listed first.
    size (2 tuple): The matplotlib figure dimensions. Defaults to (10, 10).
    t (float): The time at which to evaluate the field. Defaults to 0.

    Returns
    -------
    None
    '''

    x = np.linspace(x_lims[0], x_lims[1], nodes)
    y = np.linspace(y_lims[0], y_lims[1], nodes)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=size)

    if axis == 0:
        for i in range(nodes):
            for j in range(nodes):
                W, U[i][j], V[i][j] = field(np.array([1e-20, X[i][j], Y[i][j]]), t)
                ax.set_xlabel('$y$')
                ax.set_ylabel('$z$')
    elif axis == 1:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], W, V[i][j] = field(np.array([X[i][j], 1e-20, Y[i][j]]), t)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$z$')
    elif axis == 2:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], V[i][j], W  = field(np.array([X[i][j], Y[i][j], 1e-20]), t)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')

    color = 2 * np.log(np.hypot(U, V))
    ax.streamplot(X, Y, U, V, color=color, linewidth=1, cmap=plt.cm.jet, density=2, arrowstyle='wedge', arrowsize=1.)

    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    plt.show()
