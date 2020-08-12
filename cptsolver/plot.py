from math import ceil, floor
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import colors as col
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from cptsolver.utils import Re, inv_Re


def format_plots(dpi=150, use_tex=True, font_type='serif', font_family='Computer Modern', font_size=15, colormap='plasma'):
    '''
    Formats plots to have a cohesive style.

    Parameters
    ----------
    dpi : int, optional
        The DPI of the produced graphs. Defaults to 150.

    use_tex : bool, optional
        Whether to use Latex in the labels. Defaults to true.

    font_type : string, optional
        The font type, either 'serif' or 'sans-serif'. Defaults to 'serif'.

    font_family : string, optional
        The font family. Defaults to 'Computer Modern', which is the font used by Latex.

    font_size : float, optional
        The font size. Defaults to 15.

    colormap : string, optional
        The name of the colormap to use. Defaults to 'plasma'. See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html for a list of possibilities. 

    Returns
    -------
    None
    '''

    rc('figure', dpi=dpi)
    rc('font', **{'family': font_type, font_type: [font_family], 'size': font_size})
    rc('text', usetex=use_tex)
    rc('axes', prop_cycle=cycler(color=plt.get_cmap(colormap)(np.linspace(0, 1, 15))))
    rc('image', cmap=colormap)

    return


def plot_field(field, h_axis, v_axis, h_lim, v_lim, t=0., in_re=True, nodes=20, labels=('Horizontal Axis', 'Vertical Axis', 'Strength (T)'), title='Field Lines', fig=None, ax=None, size=(5, 5)):
    '''
    Creates a plot of the integral curves of a field along a plane passing through the origin.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a
        position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    h_axis : float[3]
        Vector lying along the horizontal axis.

    v_axis : float[3]
        Vector lying along the vertical axis.

    h_lims : float[2]
        The horizontal axis limits.

    v_lims : float[2]
        The vertical axis limits.

    t : float, optional
        The time at which to evaluate the field. Defaults to 0.

    in_re : bool, optional
        Whether the limits are in terms of Earth radii. Defaults to true.

    nodes : int, optional
        The number of field samples to use. Defaults to 20.

    labels : (string, string, string), optional
        The horizontal axis, vertical axis, and colorbar labels, respectively. Defaults to ('Horizontal Axis', 'Vertical Axis', 'Strength (T)').

    title : string, optional
        The title. Defaults to 'Field Lines'.

    fig : matplotlib figure obj, optional
        Specify if this plot is to use an external figure. Useful for creating subplots. Defaults to none.

    ax : matplotlib axis obj, optional
        Specify if this plot is to use an external set of axes. Useful for creating subplots. Defaults to none.

    size : (float, float), optional
        The size of the produced plot. Defaults to (5, 5).

    Returns
    -------
    None
    '''

    x = np.linspace(h_lim[0], h_lim[1], nodes)
    y = np.linspace(v_lim[0], v_lim[1], nodes)

    h_axis = h_axis / np.linalg.norm(h_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)
    
    for i in range(nodes):
        for j in range(nodes):
            vec_l = np.zeros(3)
            if in_re:
                vec_l = field(np.asarray((h_axis * X[i][j] + v_axis * Y[i][j]) * Re), t)
            else:
                vec_l = field(np.asarray(h_axis * X[i][j] + v_axis * Y[i][j]), t)
            U[i][j] = np.dot(vec_l, h_axis)
            V[i][j] = np.dot(vec_l, v_axis)

    color = np.hypot(U, V)

    show = False
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=size)
        show = True
    
    ax.streamplot(X, Y, U, V, color=np.log10(color), linewidth=1, cmap=plt.get_cmap(plt.rcParams['image.cmap']), density=2, arrowstyle='wedge', arrowsize=1.)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=col.LogNorm(vmin=max(np.amin(color), 1e-12), vmax=np.amax(color)), cmap=plt.get_cmap(plt.rcParams['image.cmap'])), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(labels[2], rotation=-90, va='bottom')

    ax.set_xlim(h_lim[0], h_lim[1])
    ax.set_ylim(v_lim[0], v_lim[1])

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.set_title(title)

    ax.grid()

    if show:
        plt.tight_layout()
        plt.show()

    return


def plot_distribution(history, time_ind, bins='fd', log=False, avg=False, x_lim=None, y_lim=None, labels=('Quantity', 'Number Density (AU)'), title='Distribution', fig=None, ax=None, size=(5, 5)):
    '''
    Creates histograms of the given quantity at specific instances in time. Alternatively, creates a single histogram averaged over time.

    Parameters
    ----------
    history : float[N, M]
        A history of a particular quantity. The first index denotes the particle and the second the timestep.

    time_ind : int / int[L]
        The time index for which to plot the distribution. If this is a list, it overlays L distributions unless avg is set to true.
        In this final case, it averages the distribution between time_ind[0] and time_ind[1].

    bins : int / string, optional
        The binning strategy. Can be a number for uniform binning or a string such as 'fd' or 'auto'. Defaults to 'fd'.
        See https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.hist.html for more information.

    log : bool, optional
        Whether the distribution should be shown on a logarithmic scale. Defaults to false.

    avg : bool, optional
        If time_ind is a list, averages the distribution between time_ind[0] and time_ind[1]. Defaults to false.

    x_lim : float[2], optional
        The horizontal axis limits. Defaults to none, in which case the limits are autoscaled.

    y_lim : float[2], optional
        The vertical axis limits. Defaults to none, in which case the limits are autoscaled.

    labels : (string, string), optional
        The horizontal and vertical axis labels, respectively. Defaults to ('Quantity', 'Number Density (AU)').

    title : string, optional
        The title. Defaults to 'Distribution'.

    fig : matplotlib figure obj, optional
        Specify if this plot is to use an external figure. Useful for creating subplots. Defaults to none.

    ax : matplotlib axis obj, optional
        Specify if this plot is to use an external set of axes. Useful for creating subplots. Defaults to none.

    size : (float, float), optional
        The size of the produced plot. Defaults to (5, 5).

    Returns
    -------
    None
    '''

    show = False
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=size)
        show = True

    if avg and type(time_ind) == list:
        ax.hist(np.mean(history[:, time_ind[0]:time_ind[1]], axis=1), bins=bins, histtype='stepfilled', density=True)
    else:
        ax.hist(history[:, time_ind], bins=bins, histtype='stepfilled', density=True)

    if x_lim == None:
        ax.autoscale(enable=True, tight=True, axis='x')
    else:
        ax.set_xlim(x_lim[0], x_lim[1])

    if y_lim == None:
        ax.autoscale(enable=True, tight=True, axis='y')
    else:
        ax.set_ylim(y_lim[0], y_lim[1])

    if log == True:
        ax.yscale('log', nonposy='clip')

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.set_title(title)

    ax.grid()

    if show:
        plt.tight_layout()
        plt.show()

    return


def plot_evolution(history, time, already_distribution=False, bins=100, log=True, min_val=1e-4, decimate=1, avg=False, x_lim=None, y_lim=None, x_ticks=5, y_ticks=5, labels=('Time (s)', 'Quantity', 'Number Density (AU)'), title='Evolution', fig=None, ax=None, size=(5, 5)):
    '''
    Creates a histogram of the quantity at each instant in time and shows its subsequent evolution.

    Parameters
    ----------
    history : float[N, M]
        A history of a particular quantity. The first index denotes the particle and the second the timestep.

    time : int[M]
        A time for each timestep of the history.

    already_distribution : bool, optional
        Whether the quantity is already a distribution (in the case of diffusion and transport coefficients). Defaults to false.

    bins : int, optional
        The number of bins to use when creating the distribution. Defaults to 100.

    log : bool, optional
        Whether the distribution should be shown on a logarithmic scale. Defaults to true.

    min_val : float, optional
        If the distribution is shown logarithmically, this is the minimum value the scale will use. 

    decimate : int, optional
        If set to L, displays the distribution every L timesteps. Defaults to 1.

    avg : bool, optional
        If decimate is greater than 1, averages each distribution between successive timesteps. Defaults to false.

    x_lim : float[2], optional
        The horizontal axis limits. Defaults to none, in which case the limits are autoscaled.

    y_lim : float[2], optional
        The vertical axis limits. Defaults to none, in which case the limits are autoscaled.

    x_ticks : int, optional
        The number of evenly spaced horizontal tickmarks. Defualts to 5.

    y_ticks : int, optional
        The number of evenly spaced vertical tickmarks. Defaults to 5.

    labels : (string, string, string), optional
        The horizontal axis, vertical axis, and colorbar labels, respectively. Defaults to ('Time (s)', 'Quantity', 'Number Density (AU)').

    title : string, optional
        The title. Defaults to 'Evolution'.

    fig : matplotlib figure obj, optional
        Specify if this plot is to use an external figure. Useful for creating subplots. Defaults to none.

    ax : matplotlib axis obj, optional
        Specify if this plot is to use an external set of axes. Useful for creating subplots. Defaults to none.

    size : (float, float), optional
        The size of the produced plot. Defaults to (5, 5).

    Returns
    -------
    None
    '''

    dec_hist = history[:, ::decimate]
    dec_time = time[::decimate]

    if avg:
        for i in range(np.shape(dec_hist)[1]):
            dec_hist[:, i] = np.mean(history[:, i * decimate:min((i + 1) * decimate, np.shape(history)[1])], axis=1)

    if already_distribution:
        bins = np.shape(dec_hist)[0]

    im_map = np.zeros((bins, len(dec_time)))

    for i in range(len(dec_time)):
        if already_distribution:
            im_map = dec_hist
        else:
            im_map[:, i] = np.flip(np.histogram(dec_hist[:, i], bins=np.linspace(np.amin(dec_hist), np.amax(dec_hist), num=bins + 1), density=True)[0])

    show = False
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=size)
        show = True

    if log:
        im = ax.imshow(im_map, norm=col.LogNorm(vmin=max(np.amin(im_map), min_val), vmax=np.amax(im_map)))
    else:
        im = ax.imshow(im_map)

    ax.set_xticks(np.linspace(0, len(dec_time), x_ticks))
    ax.set_yticks(np.linspace(0, bins, y_ticks))

    if x_lim == None:
        ax.set_xticklabels([f'{k:.2f}' for k in np.linspace(dec_time[0], dec_time[-1], x_ticks)])
    else:
        ax.set_xticklabels([f'{k:.2f}' for k in np.linspace(x_lim[0], x_lim[1], x_ticks)])

    if y_lim == None:
        ax.set_yticklabels([f'{k:.2f}' for k in np.linspace(np.amax(dec_hist), np.amin(dec_hist), y_ticks)])
    else:
        ax.set_yticklabels([f'{k:.2f}' for k in np.linspace(y_lim[1], y_lim[0], y_ticks)])
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.set_title(title)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(labels[2], rotation=-90, va='bottom')

    ax.grid()

    if show:
        plt.tight_layout()
        plt.show()

    return


def plot_graph(history, time, particle_ind, log=False, avg=False, x_lim=None, y_lim=None, labels=('Time (s)', 'Quantity'), title='Graph', fig=None, ax=None, size=(5, 5)):
    '''
    Plots a quantity over time for specific particles. Alternatively, plots the average of the quantity across a range of particles.

    Parameters
    ----------
    history : float[N, M]
        A history of a particular quantity. The first index denotes the particle and the second the timestep.

    time : int[M]
        A time for each timestep of the history.

    particle_ind : int / int[L]
        The particle index for which to plot the graph. If this is a list, it overlays L graphs unless avg is set to true.
        In this final case, it averages the graph over particles between particle_ind[0] and particle_ind[1].

    log : bool, optional
        Whether the graph should be shown on a logarithmic scale. Defaults to false.

    avg : bool, optional
        If particle_ind is a list, averages the graph over particles between particle_ind[0] and particle_ind[1]. Defaults to false.

    x_lim : float[2], optional
        The horizontal axis limits. Defaults to none, in which case the limits are autoscaled.

    y_lim : float[2], optional
        The vertical axis limits. Defaults to none, in which case the limits are autoscaled.

    labels : (string, string), optional
        The horizontal and vertical axis labels, respectively. Defaults to ('Time (s)', 'Quantity').

    title : string, optional
        The title. Defaults to 'Graph'.

    fig : matplotlib figure obj, optional
        Specify if this plot is to use an external figure. Useful for creating subplots. Defaults to none.

    ax : matplotlib axis obj, optional
        Specify if this plot is to use an external set of axes. Useful for creating subplots. Defaults to none.

    size : (float, float), optional
        The size of the produced plot. Defaults to (5, 5).

    Returns
    -------
    None
    '''

    show = False
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=size)
        show = True

    if type(particle_ind) == int:
        ax.plot(time, history[particle_ind])
    else:
        if avg:
            ax.plot(time, np.mean(history[particle_ind[0]:particle_ind[1]], axis=0))
        else:
            for i in particle_ind:
                ax.plot(time, history[i])

    if x_lim == None:
        ax.autoscale(enable=True, tight=True, axis='x')
    else:
        ax.set_xlim(x_lim[0], x_lim[1])

    if y_lim == None:
        ax.autoscale(enable=True, tight=True, axis='y')
    else:
        ax.set_ylim(y_lim[0], y_lim[1])

    if log == True:
        ax.set_yscale('log', nonposy='clip')

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    ax.set_title(title)

    ax.grid()

    if show:
        plt.tight_layout()
        plt.show()

    return


def plot_shared_graph(history_left, history_right, time, particle_ind, log_left=False, log_right=False, avg=False, x_lim=None, y_lim_left=None, y_lim_right=None, labels=('Time (s)', 'Left Quantity', 'Right Quantity'), title='Graph', color_separation=5, fig=None, ax=None, size=(5, 5)):
    '''
    Plots two quantities over time for specific particles. Alternatively, plots the average of the two quantities across a range of particles.  

    Parameters
    ----------
    history_left : float[N, M]
        A history of a particular quantity to use the left spine as its vertical axis. The first index denotes the particle and the second the timestep.

    history_right : float[N, M]
        A history of a particular quantity to use the right spine as its vertical axis. The first index denotes the particle and the second the timestep.

    time : int[M]
        A time for each timestep of the histories.

    particle_ind : int / int[L]
        The particle index for which to plot the graph. If this is a list, it overlays L graphs unless avg is set to true.
        In this final case, it averages the graph over particles between particle_ind[0] and particle_ind[1].

    log_left : bool, optional
        Whether the left graph should be shown on a logarithmic scale. Defaults to false.

    log_right : bool, optional
        Whether the right graph should be shown on a logarithmic scale. Defaults to false.

    avg : bool, optional
        If particle_ind is a list, averages the graph over particles between particle_ind[0] and particle_ind[1]. Defaults to false.

    x_lim : float[2], optional
        The horizontal axis limits. Defaults to none, in which case the limits are autoscaled.

    y_lim_left : float[2], optional
        The left vertical axis limits. Defaults to none, in which case the limits are autoscaled.

    y_lim_right : float[2], optional
        The right vertical axis limits. Defaults to none, in which case the limits are autoscaled.

    labels : (string, string, string), optional
        The horizontal and two vertical axis labels, respectively. Defaults to ('Time (s)', 'Quantity').

    title : string, optional
        The title. Defaults to 'Graph'.

    color_separation : int, optional
        The difference in color cycles between the left and right graphs. Defaults to 5.

    fig : matplotlib figure obj, optional
        Specify if this plot is to use an external figure. Useful for creating subplots. Defaults to none.

    ax : matplotlib axis obj, optional
        Specify if this plot is to use an external set of axes. Useful for creating subplots. Defaults to none.

    size : (float, float), optional
        The size of the produced plot. Defaults to (5, 5).

    Returns
    -------
    None
    '''

    show = False
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=size)
        show = True

    if type(particle_ind) == int:
        ax.plot(time, history_left[particle_ind])
    else:
        if avg:
            ax.plot(time, np.mean(history_left[particle_ind[0]:particle_ind[1]], axis=0))
        else:
            for i in particle_ind:
                ax.plot(time, history_left[i])

    if y_lim_left == None:
        ax.autoscale(enable=True, tight=True, axis='y')
    else:
        ax.set_ylim(y_lim_left[0], y_lim_left[1])

    if log_left == True:
        ax.set_yscale('log', nonposy='clip')


    ax.spines['left'].set_color(f'C0')
    ax.yaxis.label.set_color(f'C0')
    ax.tick_params(axis='y', colors=f'C0')
    ax.set_ylabel(labels[1], color=f'C0')

    ax.grid(color='C0', alpha=0.4)

    ax2 = ax.twinx()
    
    if type(particle_ind) == int:
        ax2.plot(time, history_right[particle_ind], color=f'C{color_separation}')
    else:
        if avg:
            ax2.plot(time, np.mean(history_right[particle_ind[0]:particle_ind[1]], axis=0), color=f'C{color_separation}')
        else:
            for i, j in enumerate(particle_ind):
                ax2.plot(time, history_right[j], color=f'C{i + color_separation}')

    if y_lim_right == None:
        ax2.autoscale(enable=True, tight=True, axis='y')
    else:
        ax2.set_ylim(y_lim_right[0], y_lim_right[1])

    if log_right == True:
        ax2.set_yscale('log', nonposy='clip')


    ax2.spines['left'].set_color(f'C0')
    ax2.spines['right'].set_color(f'C{color_separation}')
    ax2.yaxis.label.set_color(f'C{color_separation}')
    ax2.tick_params(axis='y', colors=f'C{color_separation}')
    ax2.set_ylabel(labels[2], color=f'C{color_separation}')

    ax2.grid(color=f'C{color_separation}', alpha=0.4)

    if x_lim == None:
        ax.autoscale(enable=True, tight=True, axis='x')
        ax2.autoscale(enable=True, tight=True, axis='x')
    else:
        ax.set_xlim(x_lim[0], x_lim[1])
        ax2.set_xlim(x_lim[0], x_lim[1])

    ax.set_xlabel(labels[0])

    ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return


def plot_3d(history, particle_ind, elev, azim, x_lim=None, y_lim=None, z_lim=None, labels=(r'$x_{GSM}$ ($R_E$)', r'$y_{GSM}$ ($R_E$)', r'$z_{GSM}$ ($R_E$)'), title='Graph', fig=None, ax=None, size=(5, 5)):
    '''
    Plots a 3D quantity.

    Parameters
    ----------
    history : float[N, M, 3]
        A history of a particular 3D quantity. The first index denotes the particle, the second the timestep, and the third the dimension.

    particle_ind : int / int[L]
        The particle index for which to plot the graph. If this is a list, it overlays L graphs.

    elev : float
        The elevation or altitude angle of the viewport (in deg).

    azim : float
        The azimuthal angle of the viewport (in deg).

    x_lim : float[2], optional
        The x-axis limits. Defaults to none, in which case the limits are autoscaled.

    y_lim : float[2], optional
        The y-axis limits. Defaults to none, in which case the limits are autoscaled.

    z_lim : float[2], optional
        The z-axis limits. Defaults to none, in which case the limits are autoscaled.

    labels : (string, string, string), optional
        The x-axis, y-axis, and z-axis labels, respectively. Defaults to (r'$x_{GSM}$ ($R_E$)', r'$y_{GSM}$ ($R_E$)', r'$z_{GSM}$ ($R_E$)').

    title : string, optional
        The title. Defaults to 'Graph'.

    fig : matplotlib figure obj, optional
        Specify if this plot is to use an external figure. Useful for creating subplots. Defaults to none.

    ax : matplotlib axis obj, optional
        Specify if this plot is to use an external set of axes. Useful for creating subplots. Defaults to none.

    size : (float, float), optional
        The size of the produced plot. Defaults to (5, 5).

    Returns
    -------
    None
    '''

    show = False
    if fig == None or ax == None:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')
        show = True

    if type(particle_ind) == int:
        traj = history[particle_ind, :]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
    else:
        for i in particle_ind:
            traj = history[i, :]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])

    if x_lim == None:
        ax.autoscale(enable=True, tight=True, axis='x')
    else:
        ax.set_xlim(x_lim[0], x_lim[1])

    if y_lim == None:
        ax.autoscale(enable=True, tight=True, axis='y')
    else:
        ax.set_xlim(y_lim[0], y_lim[1])

    if z_lim == None:
        ax.autoscale(enable=True, tight=True, axis='z')
    else:
        ax.set_xlim(z_lim[0], z_lim[1])

    ax.set_xlabel(labels[0], linespacing=1.5, labelpad=10)
    ax.set_ylabel(labels[1], linespacing=3, labelpad=10)
    ax.set_zlabel(labels[2], linespacing=3, labelpad=10)

    ax.set_title(title)

    ax.view_init(elev, azim)

    ax.grid()

    if show:
        plt.tight_layout()
        plt.show()

    return
