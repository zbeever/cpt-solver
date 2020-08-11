from math import ceil, floor
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import colors as col
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from utils import Re, inv_Re


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
    rc('axes', prop_cycle=cycler(color=plt.get_cmap(colormap)(np.linspace(0, 1, 10))))
    rc('image', cmap=colormap)

    return


def plot_field(field, h_axis, v_axis, x_lims, y_lims, t=0., gsm=True, nodes=20, labels=('x ($R_E$)', 'z ($R_E$)'), title='Field Lines', fig=None, ax=None, size=(5, 5)):
    '''
    Creates a plot of the integral curves of a field along a plane passing through the origin.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a
        position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    h_axis : float[3]

    v_axis : float[3]

    x_lims : float[2]

    y_lims : float[2]

    t : float, optional

    gsm : bool, optional

    nodes : int, optional

    labels : (string, string), optional

    title : string, optional

    fig : matplotlib figure obj, optional

    ax : matplotlib axis obj, optional

    size : (float, float), optional

    Returns
    -------
    None
    '''

    x = np.linspace(x_lims[0], x_lims[1], nodes)
    y = np.linspace(y_lims[0], y_lims[1], nodes)

    h_axis = h_axis / np.linalg.norm(h_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)
    
    for i in range(nodes):
        for j in range(nodes):
            vec_l = np.zeros(3)
            if not gsm:
                vec_l = field(np.asarray(h_axis * X[i][j] + v_axis * Y[i][j]), t)
            else:
                vec_l = field(np.asarray((h_axis * X[i][j] + v_axis * Y[i][j]) * Re), t)
            U[i][j] = np.dot(vec_l, h_axis)
            V[i][j] = np.dot(vec_l, v_axis)

    color = np.hypot(U, V)

    show = False
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=size)
        show = True
    
    ax.streamplot(X, Y, U, V, color=np.log10(color), linewidth=1, cmap=plt.get_cmap(plt.rcParams['image.cmap']), density=2, arrowstyle='wedge', arrowsize=1.)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=col.LogNorm(vmin=max(np.amin(color), 1e-12), vmax=np.amax(color)), cmap=plt.get_cmap(plt.rcParams['image.cmap'])), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Strength (T)', rotation=-90, va='bottom')

    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])

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
    Plots two quantity over time for specific particles. Alternatively, plots the average of the two quantities across a range of particles.  

    Parameters
    ----------

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
