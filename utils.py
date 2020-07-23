import numpy as np
from numba import njit
from diagnostics import *
import scipy.constants as sp
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

axis_num = {
            'x': 0,
            'y': 1,
            'z': 2
           }

Re = 6.371e6      # m
inv_Re = 1. / Re  # m^-1


def format_bytes(size):
    '''
    Utility function to format an integer number of bytes to a human-readable format.

    Parameters
    ==========
    size (int): Number of bytes.

    Returns
    =======
    size (float): Rescaled size of the data.
    power_label (string): The associated unit.
    '''

    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}

    while size > power:
        size /= power
        n += 1

    return size, power_labels[n] + 'bytes'


@njit
def J_to_eV(E):
    '''
    Converts joules to electronvolts.

    Parameters
    ==========
    E (float): An energy (in joules).

    Returns
    =======
    E (float): An energy (in electronvolts).
    '''

    return 1.0 / sp.e * E


@njit
def eV_to_J(E):
    '''
    Converts electronvolts to joules.

    Parameters
    ==========
    E (float): An energy (in electronvolts).

    Returns
    =======
    E (float): An energy (in joules).
    '''

    return sp.e * E


@njit
def dot(v, w):
    '''
    Dots two vectors. The reason for this (over np.dot) is to avoid the slowdown that comes
    when using np.dot in a Numba function on vectors whose components are not contiguous in memory.

    Parameters
    ==========
    v (3x1 numpy array): First vector.
    w (3x1 numpy array): Second vector.

    Returns
    =======
    v_dot_w (float): The dot product of v and w. 
    '''

    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]


@njit
def gamma(v):
    '''
    Calculates the standard relativistic factor from a SI velocity vector.

    Parameters
    ==========
    v (3x1 numpy array): The velocity vector (in m/s).

    Returns
    =======
    gamma (float): The relativistic factor.
    '''

    return 1.0 / np.sqrt(1 - dot(v, v) / sp.c**2)


@njit
def local_onb(r, b_field, t=0.):
    '''
    Constructs an orthonormal basis at a given location with the z axis along the magnetic field and the x axis directed toward the origin.

    Parameters
    ==========
    r (3x1 numpy array): The origin of the new coordinate system.
    b_field(r, t): The field function (this is obtained through the currying functions in fields.py).
    t0 (float): The universal time (in seconds). Defaults to 0.

    Returns
    =======
    local_x (3x1 numpy array): Local x axis unit vector.
    local_y (3x1 numpy array): Local y axis unit vector.
    local_z (3x1 numpy array): Local z axis unit vector.
    '''

    B = b_field(r, t)

    local_z = B
    if np.dot(local_z, local_z) == 0:
        local_z = np.array([0., 0., 1.])
    else:
        local_z = local_z / np.linalg.norm(local_z)

    local_x = -r
    local_x = local_x - dot(local_x, local_z) * local_z
    if np.dot(local_x, local_x) == 0:
        local_x = np.array([-1., 0., 0.])
    else:
        local_x = local_x / np.linalg.norm(local_x)

    local_y = np.cross(local_z, local_x)
    return local_x, local_y, local_z


@njit
def velocity_vec(r, K, m, b_field, pitch_angle, phase_angle, t=0.):
    '''
    Generates a velocity vector from a particle's energy, pitch angle, and phase angle.

    Parameters
    ==========
    r (3x1 numpy array): The location (in m) of the particle.
    K (float): The kinetic energy (in eV) of the particle.
    m (float): The mass (in kg) of the particle.
    b_field(r, t): The field function (this is obtained through the currying functions in fields.py).
    pitch_angle (float): The angle the velocity vector makes with respect to the magnetic field (in radians).
    phase_angle (float): The angle the velocity vector makes with respect to (r x B) x B (in radians).
    t0 (float): The universal time (in seconds). Defaults to 0.

    Returns
    =======
    v (3x1 numpy array): The velocity (in m/s) of the particle.
    '''

    local_x, local_y, local_z = local_onb(r, b_field, t)

    v_dir = np.sin(pitch_angle) * np.cos(phase_angle) * local_x + np.sin(pitch_angle) * np.sin(phase_angle) * local_y + np.cos(pitch_angle) * local_z

    gamma_v = eV_to_J(K) / (m * sp.c ** 2.0) + 1.0
    v_mag = sp.c * np.sqrt(1. - gamma_v ** (-2.0))

    if np.dot(v_dir, v_dir) == 0.0:
        return np.array([0.0, 0.0, 0.0])
    else:
        return v_dir / np.linalg.norm(v_dir) * v_mag


@njit
def grad(field, r, eps=1e-6):
    '''
    Numerically finds the gradient of a field at a given point.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    r (3x1 numpy array): The location at which to find the gradient.
    eps (float): The size of epsilon for use in the definition of the partial derivative. Defaults to 1e-6.

    Returns
    =======
    grad (3x1 numpy array): The gradient of the field at the given point.
    '''

    x_offset = np.array([eps, 0., 0.])
    y_offset = np.array([0., eps, 0.])
    z_offset = np.array([0, 0., eps])

    grad = np.zeros(3)
    grad[0] = (np.linalg.norm(field(r + x_offset)) - np.linalg.norm(field(r - x_offset))) / (2 * eps)
    grad[1] = (np.linalg.norm(field(r + y_offset)) - np.linalg.norm(field(r - y_offset))) / (2 * eps)
    grad[2] = (np.linalg.norm(field(r + z_offset)) - np.linalg.norm(field(r - z_offset))) / (2 * eps)

    return grad


@njit
def jacobian(field, r, eps=1e-6):
    '''
    Numerically finds the Jacobian of a field at a given point.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    r (3x1 numpy array): The location at which to find the Jacobian.
    eps (float): The size of epsilon for use in the definition of the partial derivative. Defaults to 1e-6.

    Returns
    =======
    jac (3x3 numpy array): The gradient of the field at the given point.
    '''

    x_offset = np.array([eps, 0., 0.])
    y_offset = np.array([0., eps, 0.])
    z_offset = np.array([0, 0., eps])

    jac = np.zeros((3, 3))
    jac[:, 0] = (field(r + x_offset) - field(r - x_offset)) / (2 * eps)
    jac[:, 1] = (field(r + y_offset) - field(r - y_offset)) / (2 * eps)
    jac[:, 2] = (field(r + z_offset) - field(r - z_offset)) / (2 * eps)

    return jac


@njit
def flc(field, r, eps=1e-6):
    '''
    Numerically finds the radius of curvature of the field line at a given point.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    r (3x1 numpy array): The location at which to find the field line curvature.
    eps (float): The size of epsilon for use in the definition of the partial derivative. Defaults to 1e-6.

    Returns
    =======
    Rc (float): The radius of curvature (in m).
    '''

    x_offset = np.array([eps, 0.0, 0.0])
    y_offset = np.array([0.0, eps, 0.0])
    z_offset = np.array([0.0, 0.0, eps])
    
    b = field(r)
    b /= np.linalg.norm(b)

    fx1 = field(r + x_offset)
    fx1 /= np.linalg.norm(fx1)
    
    fx0 = field(r - x_offset)
    fx0 /= np.linalg.norm(fx0)
    
    fy1 = field(r + y_offset)
    fy1 /= np.linalg.norm(fy1)
    
    fy0 = field(r - y_offset)
    fy0 /= np.linalg.norm(fy0)
    
    fz1 = field(r + z_offset)
    fz1 /= np.linalg.norm(fz1)
    
    fz0 = field(r - z_offset)
    fz0 /= np.linalg.norm(fz0)

    J = np.zeros((3, 3))
    J[:, 0] = (fx1 - fx0) / (2 * eps)
    J[:, 1] = (fy1 - fy0) / (2 * eps)
    J[:, 2] = (fz1 - fz0) / (2 * eps)
    
    return (1.0 / np.linalg.norm(np.dot(J, b)))


@njit
def field_line(field, r, tol):
    '''
    Traces the field line of a given magnetosphere model using the Runge-Kutta-Fehlberg method. Use this for the Tsyganenko and IGRF models.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    r (3x1 numpy array): A point which the field line runs through.
    eps (float): The maximum error allowed by the solver.

    Returns
    =======
    rr (Nx3 numpy array): An array of points marking the field line. Begins and stops where the field line is 0.5 R_E away from the origin.
    '''

    def rk45_step(field, r, h, tol, direction):
        a = np.sign(direction)
        
        k1 = a * field(r)
        k1 /= np.linalg.norm(k1)
        k1 *= h

        k2 = a * field(r + 0.25 * k1)
        k2 /= np.linalg.norm(k2)
        k2 *= h

        k3 = a * field(r + 0.09375 * k1 + 0.28125 * k2)
        k3 /= np.linalg.norm(k3)
        k3 *= h

        k4 = a * field(r + 0.87938097405553 * k1 - 3.2771961766045 * k2 + 3.3208921256259 * k3)
        k4 /= np.linalg.norm(k4)
        k4 *= h

        k5 = a * field(r + 2.0324074074074 * k1 - 8 * k2 + 7.1734892787524 * k3 - 0.20589668615984 * k4)
        k5 /= np.linalg.norm(k5)
        k5 *= h

        k6 = a * field(r - 0.2962962962963 * k1 + 2 * k2 - 1.3816764132554 * k3 + 0.45297270955166 * k4 - 0.275 * k5)
        k6 /= np.linalg.norm(k6)
        k6 *= h

        y_plus_1 = r + 0.11574074074074 * k1 + 0.54892787524366 * k3 + 0.53533138401559 * k4 - 0.2 * k5
        z_plus_1 = r + 0.11851851851852 * k1 + 0.51898635477583 * k3 + 0.50613149034202 * k4 - 0.18 * k5 + 0.036363636363636 * k6

        t_plus_1 = z_plus_1 - y_plus_1
        h = 0.9 * h * min(max(np.sqrt(tol / (2 * np.linalg.norm(t_plus_1))), 0.3), 2)
        
        return z_plus_1, h

    rrb = np.zeros((1, 3))
    rrb[0] = r
    
    h = 1e4
    while True:
        r, h = rk45_step(field, r, h, tol, -1)

        if np.linalg.norm(r) <= 0.5 * Re:
            break

        k = np.zeros((1, 3))
        k[0] = r
        rrb = np.append(rrb, k, axis=0)
        
    r = np.copy(rrb[0])
    
    rrf = np.zeros((1, 3))
    rrf[0] = r
    
    while True:
        r, h = rk45_step(field, r, h, tol, 1)

        if np.linalg.norm(r) <= 0.5 * Re:
            break

        k = np.zeros((1, 3))
        k[0] = r
        rrf = np.append(rrf, k, axis=0)
        
    rr = np.append(rrf[::-1], rrb, axis=0)
    
    return rr


@njit
def b_along_path(field, rr):
    '''
    Gives the magnetic field along a path.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    rr (Nx3 numpy array): An array of points marking the path.

    Returns
    =======
    field_vec (Nx3 numpy array): The field vector at each point along the path.
    field_mag (Nx1 numpy array): The strength of the field along the path.
    field_rad_mag (Nx1 numpy array): The strength of the radial (x and y) field along the path.
    '''

    steps = len(rr[:, 0])
    field_vec = np.zeros((steps, 3))
    field_mag = np.zeros(steps)
    field_rad_mag = np.zeros(steps)
    
    for i in range(steps):
        vec = field(rr[i])
        
        field_vec[i] = vec
        field_mag[i] = np.linalg.norm(vec)
        field_rad_mag[i] = np.sqrt(vec[0]**2 + vec[1]**2)
        
    return field_vec, field_mag, field_rad_mag


@njit
def field_reversal(field, rr):
    '''
    Finds the point along a path at which the radial magnitude of the field is at a minimum.
    For a field with no dipole tilt, this coincides with the location of maximum curvature along a field line.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    rr (Nx3 numpy array): An array of points marking the path.

    Returns
    =======
    r (3x1 numpy array): The point of minimum radial field strength.
    '''

    b_vec, b_mag, b_rad_mag = b_along_path(field, rr)
    return rr[b_rad_mag.argmin()]


@njit
def adiabaticity(field, rr, K, m=sp.m_e, q=-sp.e):
    '''
    For a given particle, returns the adiabaticity along a path. This is usually referred to by kappa = sqrt(R_c/rho),
    where R_c is the radius of curvature of the field line at that point and rho is the particle's gyroradius.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    rr (Nx3 numpy array): An array of points marking the path.
    K (float): The kinetic energy of the particle (in eV).
    m (float): The mass of the particle (in kg). Defaults to the electron mass.
    q (float): The electric charge of the particle (in C). Defaults to the electron charge (the negative unit charge).

    Returns
    =======
    kappa (Nx1 numpy array): The adiabaticity of each point along the path.
    '''

    hist_new = np.zeros((len(rr[:, 0])))

    gamma_v = 1 + eV_to_J(K) / (m * sp.c**2)
    v = (sp.c / gamma_v) * np.sqrt(gamma_v**2 - 1)

    for i in range(len(rr[:, 0])):
        b = field(rr[i, :])

        rho_0 = gamma_v * m * v / (abs(q) * np.linalg.norm(b))
        
        R_c = flc(field, rr[i, :])
        hist_new[i] = rho_0 / R_c

    return hist_new


def plot_field(field, axis, nodes, x_lims, y_lims, size=(10, 10), t=0.):
    '''
    Creates a plot of the integral curves of a field along one of the three axis-aligned planes passing through the origin.

    Parameters
    ==========
    field(r, t): The field function (this is obtained through the currying functions in fields.py).
    axis (string): Either 'x', 'y', or 'z.' This is the axis the plane of the graph will be orthogonal to.
    nodes (int): The number of samples to be taken along each axis.
    x_lims (2 list): A 2 element list consisting of the horizontal axis limits. The lower value is listed first.
    y_lims (2 list): A 2 element list consisting of the vertical axis limits. The lower value is listed first.
    size (2 tuple): The matplotlib figure dimensions. Defaults to (10, 10).
    t (float): The time at which to evaluate the field. Defaults to 0.

    Returns
    =======
    None
    '''

    x = np.linspace(x_lims[0], x_lims[1], nodes)
    y = np.linspace(y_lims[0], y_lims[1], nodes)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=size)

    if axis_num[axis] == 0:
        for i in range(nodes):
            for j in range(nodes):
                W, U[i][j], V[i][j] = field(np.array([1e-20, X[i][j], Y[i][j]]), t)
                ax.set_xlabel('$y$')
                ax.set_ylabel('$z$')
    elif axis_num[axis] == 1:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], W, V[i][j] = field(np.array([X[i][j], 1e-20, Y[i][j]]), t)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$z$')
    elif axis_num[axis] == 2:
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
