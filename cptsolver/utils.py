import numpy as np
from datetime import datetime
from calendar import monthrange
import scipy.constants as sp
from scipy.io import loadmat
from numba import njit, prange
from math import sqrt
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

Re = 6.371e6     # m
inv_Re = 1. / Re # m^-1


def normalized_bounce_time(field, L, steps=200, tol=1e-6, max_iter=5000, window=51, polyord=2):
    '''
    Returns the normalized bounce time along the midnight-plane field line (specified by L) of a magnetosphere model.

    Parameters
    ----------
    field(r, t=0.) : function
        The magnetosphere model. See fields.py.

    L : float
        The L-shell value.

    steps : int, optional
        The number of different equatorial pitch angles to examine. Defaults to 200.

    tol : float, optional
        The tolerance of the field line tracer. Defaults to 1e-6.

    max_iter : int, optional
        The maximum iterations of the field line tracer. Defaults to 5000.

    window : int, optional
        The window used by the Savitzky-Golay filter. Defaults to 51.

    polyord : int, optional
        The order of polynomial used by the Savitzky-Golay filter.

    Returns
    -------
    T(alpha_eq) : function
        The normalized bounce time function.
    '''

    rr = field_line(field, np.array([-L * Re, 0., 0.]), 0, tol, max_iter)
    bv, bm, brm = b_along_path(field, rr)
    cs_ind = bm.argmin()
    
    r_eq = np.linalg.norm(rr[cs_ind])
    b0 = bm[cs_ind]
    
    alpha_eqs = np.radians(np.linspace(0, 90, steps))
    Ts = np.zeros_like(alpha_eqs)
    
    for j, alpha_eq in enumerate(alpha_eqs):
        upward_integral = 0
        i = cs_ind

        ds = np.linalg.norm(rr[i + 1] - rr[i])
        Bs = np.linalg.norm(field(rr[i + 1]))
        arcsin_arg = np.sqrt(Bs / b0) * np.sin(alpha_eq)

        while arcsin_arg <= 1 and i < len(bm) - 2:
            i += 1

            upward_integral += ds / (np.cos(np.arcsin(np.sqrt(Bs / b0) * np.sin(alpha_eq))))

            ds = np.linalg.norm(rr[i + 1] - rr[i])
            Bs = np.linalg.norm(field(rr[i + 1]))
            arcsin_arg = np.sqrt(Bs / b0) * np.sin(alpha_eq)

        downward_integral = 0
        i = cs_ind

        ds = np.linalg.norm(rr[i - 1] - rr[i])
        Bs = np.linalg.norm(field(rr[i - 1]))
        arcsin_arg = np.sqrt(Bs / b0) * np.sin(alpha_eq)

        while arcsin_arg <= 1 and i - 1 > 0:
            i -= 1

            downward_integral += ds / (np.cos(np.arcsin(np.sqrt(Bs / b0) * np.sin(alpha_eq))))

            ds = np.linalg.norm(rr[i - 1] - rr[i])
            Bs = np.linalg.norm(field(rr[i - 1]))
            arcsin_arg = np.sqrt(Bs / b0) * np.sin(alpha_eq)
        
        Ts[j] = 0.5 * (upward_integral + downward_integral) / r_eq
        
    smoothed_Ts = savgol_filter(Ts, window, polyord)
    
    def T(alpha_eq):
        return np.interp(alpha_eq, alpha_eqs, smoothed_Ts)
    
    return np.vectorize(T)


def harris_kappa(E, m, q, alpha_eq, b0x, sigma, L_cs):
    '''
    Returns the kappa parameter for a particle in a given Harris sheet.

    Parameters
    ----------
    E : float
        The kinetic energy of the particle (in eV).

    m : float
        The mass of the particle (in kg).

    q : float
        The charge of the particle (in C).

    alpha_eq : float
        The equatorial pitch angle of the particle (in radians).

    b0x : float
        The b0x parameter (in T) of the associated Harris model. The minimum B magnitude is given by sigma * b0x.

    sigma : float
        The sigma parameter of the associated Harris model. This is the ratio of b0z / b0x.

    L_cs : float
        The L_cs parameter (in m) of the associated Harris model. This is the current sheet thickness.

    Returns
    -------
    kappa : float
        The kappa parameter, characterizing the particle's adiabaticity.
    '''

    K = eV_to_J(E)
    gamma = K / (m * sp.c**2) + 1.
    v = c * np.sqrt(1. - 1. / gamma**2)
    kappa = np.sqrt(np.abs(q) * sigma**2 * L_cs * b0x / (gamma * m * v * np.sin(alpha_eq)))
    return kappa


@njit
def gyrovector(v, b, m, q):
    '''
    Given a velocity and B vector (assumed to be at the guiding center), returns the gyrovector
    stretching from the guiding center to the particle's position.

    Parameters
    ----------
    v : float[3]
        The velocity of the particle in m/s.

    b : float[3]
        The magnetic field vector in T. This is assumed to be at the guiding center.

    m : float
        The mass of the particle in kg.

    q : float
        The charge of the particle in C.

    Returns
    -------
    rho : float[3]
        The gyrovector in m.
    '''

    gamma = 1. / np.sqrt(1. - np.dot(v, v) / sp.c**2)  
    return gamma * m / (q * np.dot(b, b)) * np.cross(b, v)


@njit
def guiding_center(r, v, field, m, q, tol=1e-3, max_iter=20):
    '''
    Attempts to iteratively find the guiding center of a particle given its instantaneous position.

    Parameters
    ----------
    r : float[3]
        The instanteous position of the particle in m.
    v : float[3]
        The velocity of the particle in m/s.

    field(r, t=0.) : function
        The magnetic field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    m : float
        The mass of the particle in kg.

    q : float
        The charge of the particle in C.

    tol : float, optional
        The tolerance of the solver. This is the maximum difference allowed between successive solutions. Defaults to 1e-3.

    max_iter : int, optional
        The maximum number of iterations for the solver to run. Defaults to 20.

    Returns
    -------
    gc1 : float[3]
        The guiding center of the particle in m.

    b : float[3]
        The magnetic field (in T) at the guiding center.
    '''
    b = field(r)
    gc0 = r - gyrovector(v, b, m, q)

    i = 1
    while i <= max_iter:
        b = field(gc0)
        gc1 = r - gyrovector(v, b, m, q)

        if np.linalg.norm(gc1 - gc0) / np.linalg.norm(gc0) < tol:
            b = field(gc1)
            return gc1, b

        gc0 = gc1
        i += 1


def harris_params_from_txx(field, L, L_cs, t=0., tol=1e-5, eps=1e-1):
    '''
    Given an L shell, a magnetic field model, and an assumed current sheet thickness, generates Harris current sheet parameters.

    Parameters
    ----------
    field(r, t=0.) : function
        The magnetic field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    L : float
        The L-shell.

    L_cs : float
        The current sheet thickness (in m).

    t : float, optional
        The time after the initial time (in s). Defaults to 0.

    tol : float, optional
        The tolerance to use in the field line tracer. Defaults to 1e-5.

    eps : float, optional
        The value of epsilon to use in the field line curvature function. Defaults to 1e-1.

    Returns
    -------
    b0x : float
        The maximum value of the x-component of the magnetic field.

    sigma : float
        The ratio of b0z / b0x.

    L_cs : float
        The current sheet thickness (in m).
    '''

    r = np.array([-L * Re, 0., 0.])
    rr = field_line(field, r, t, tol)
    bv, bm, brm = b_along_path(field, rr)
    cs_ind = bm.argmin()

    R_c = flc(field, rr[cs_ind], t, eps)
    sigma = R_c / L_cs
    b0x = bv[cs_ind, 2] / sigma

    return b0x, sigma, L_cs


@njit
def harris_params_from_txx_guess_sigma(field, L, t=0., tol=1e-5, eps=1e-1):
    '''
    Given an L shell and a magnetic field model this function generates Harris current sheet parameters.
    It estimates a sigma by averaging the values of sigma at the local minima flanking the current sheet.
    This is experimental and not guaranteed to produce physical results.

    Parameters
    ----------
    field(r, t=0.) : function
        The magnetic field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    L : float
        The L-shell.

    t : float, optional
        The time after the initial time (in s). Defaults to 0.

    tol : float, optional
        The tolerance to use in the field line tracer. Defaults to 1e-5.

    eps : float, optional
        The value of epsilon to use in the field line curvature function. Defaults to 1e-1.

    Returns
    -------
    b0x : float
        The maximum value of the x-component of the magnetic field.

    sigma : float
        The ratio of b0z / b0x.

    L_cs : float
        The current sheet thickness (in m).
    '''

    r = np.array([-L * Re, 0., 0.])
    rr = field_line(field, r, t, tol)
    bv, bm, brm = b_along_path(field, rr)
    cs_ind = bm.argmin()

    R_c = flc(field, rr[cs_ind], t, eps)

    sigmas = np.abs(bv[:, 2] / bv[:, 0])
    local_minima = np.argwhere(np.r_[True, sigmas[1:] < sigmas[:-1]] & np.r_[sigmas[:-1] < sigmas[1:], True])[:, 0]
    left = np.argwhere(np.diff(np.sign(local_minima - cs_ind)))[:, 0][0]
    right = left + 1
    sigma = (sigmas[local_minima[left]] + sigmas[local_minima[right]]) * 0.5

    L_cs = R_c / sigma
    b0x = bv[cs_ind, 2] / sigma

    return b0x, sigma, L_cs


def get_txx_params(qin_denton_filename, omni_filename, timestamp, model='t04'):
    '''
    Given the 1-minute datasets available from http://virbo.org/QinDenton and https://omniweb.gsfc.nasa.gov/form/omni_min.html,
    parses the data for input to the Tsyganenko models.

    Parameters
    ----------
    qin_denton_filename : string
        The path to the Qin-Denton file. This should be the 1 minute increment dataset. Do not include the extension. The format should be .mat.

    omni_filename : string
        The path to the OMNI files. This should include the x, y, and z components of the solar wind velocity. Do not include the extension. Their formats should be .lst and .fmt.

    timestamp : datetime obj
        The data and time to consider.

    model : string, optional
        The model for which to format the parameters. Use either 't89', 't96', 't01', or 't04'. Defaults to 't04'.

    Returns
    -------
    parmod : int / float[10]
        The parameter array to be input into the Tsyganenko model. For the T89 model this is an integer. For all the others this is a 10 element list of floats.

    ut : int
        The time in seconds since 1/1/1970.

    v_sw : float[3]
        The velocity vector for the solar wind.

    '''

    data = loadmat(qin_denton_filename + '.mat')
    prev_days = 0

    for i in range(timestamp.month - 1):
        prev_days += monthrange(timestamp.year, i + 1)[1]

    doy = prev_days + timestamp.day

    ind_for_year = np.argwhere(data['Year'] == timestamp.year)[:, 0]
    ind_for_month = np.argwhere(data['DOY'][ind_for_year] == doy)[:, 0]
    ind_for_hour = np.argwhere(data['hour'][ind_for_year][ind_for_month] == timestamp.hour)[:, 0]
    ind_for_min = ind_for_year[ind_for_month][ind_for_hour][timestamp.minute]

    t0 = datetime(1970, 1, 1)
    ut = (timestamp - t0).total_seconds()

    v_sw_data = open(omni_filename + '.lst')
    v_sw_form = open(omni_filename + '.fmt')

    v_inds = [-1, -1, -1]

    for line in v_sw_form:
        for i, dim in enumerate(['x', 'y', 'z']):
            is_line = line.find(f'V{dim} Velocity')
            if is_line >= 0:
                v_inds[i] = int(line.split()[0]) - 1
                break

    v_sw_form.close()
    v_sw = np.zeros(3)

    for line in v_sw_data:
        line_data = line.split()
        
        year   = int(line_data[0])
        line_doy    = int(line_data[1])
        hour   = int(line_data[2])
        minute = int(line_data[3])
        
        if year == timestamp.year and line_doy == doy and hour == timestamp.hour and minute == timestamp.minute:
            v_sw[0] = float(line_data[v_inds[0]])
            v_sw[1] = float(line_data[v_inds[1]])
            v_sw[2] = float(line_data[v_inds[2]])
            break

    v_sw_data.close()

    kp    = int(round(data['Kp'][ind_for_min][0]))

    pdyn  = data['Pdyn'][ind_for_min][0]
    dst   = data['Dst'][ind_for_min][0]
    byimf = data['ByIMF'][ind_for_min][0]
    bzimf = data['BzIMF'][ind_for_min][0]

    g1    = data['G1'][ind_for_min][0]
    g2    = data['G2'][ind_for_min][0]

    w1    = data['W1'][ind_for_min][0]
    w2    = data['W2'][ind_for_min][0]
    w3    = data['W3'][ind_for_min][0]
    w4    = data['W4'][ind_for_min][0]
    w5    = data['W5'][ind_for_min][0]
    w6    = data['W6'][ind_for_min][0]

    if model == 't89':
        return kp, ut, v_sw
    elif model == 't96':
        parmod = np.array([pdyn, dst, byimf, bzimf, 0., 0., 0., 0., 0., 0.])
        return parmod, ut, v_sw
    elif model == 't01':
        parmod = np.array([pdyn, dst, byimf, bzimf, g1, g2, 0., 0., 0., 0.])
        return parmod, ut, v_sw
    elif model == 't04':
        parmod = np.array([pdyn, dst, byimf, bzimf, w1, w2, w3, w4, w5, w6])
        return parmod, ut, v_sw
    else:
        raise NameError('Model type not recognized. Use t89/t96/t01/t04.')
        

def format_bytes(size):
    '''
    Utility function to format an integer number of bytes to a human-readable format.

    Parameters
    ----------
    size : int
        Number of bytes.

    Returns
    -------
    size : float
        Rescaled size of the data.

    power_label : string
        The associated unit (e.g. megabyte).
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
    ----------
    E : float
        An energy (in joules).

    Returns
    -------
    E : float
        An energy (in electronvolts).
    '''

    return 1.0 / sp.e * E


@njit
def eV_to_J(E):
    '''
    Converts electronvolts to joules.

    Parameters
    ----------
    E : float
        An energy (in electronvolts).

    Returns
    -------
    E : float
        An energy (in joules).
    '''

    return sp.e * E


@njit
def dot(v, w):
    '''
    Dots two vectors. The reason for this (over np.dot) is to avoid the slowdown that comes
    when using np.dot in a Numba function on vectors whose components are not contiguous in memory.

    Parameters
    ----------
    v : float[3]
        First vector.

    w : float[3]
        Second vector.

    Returns
    -------
    v_dot_w : float
        The dot product of v and w. 
    '''

    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]


@njit
def cross(v, w):
    '''
    Crosses two vectors. The reason for this (over np.cross) is to avoid the slowdown that comes
    when using np.dot in a Numba function on vectors whose components are not contiguous in memory.

    Parameters
    ----------
    v : float[3]
        First vector.
    w : float[3]
        Second vector.

    Returns
    -------
    v_cross_w : float[3]
        The cross product of v and w. 
    '''
    u = np.zeros(3)

    u[0] = v[1] * w[2] - v[2] * w[1]
    u[1] = v[2] * w[0] - v[0] * w[2]
    u[2] = v[0] * w[1] - v[1] * w[0]

    return u


@njit
def gamma(v):
    '''
    Calculates the standard relativistic factor from a velocity vector.

    Parameters
    ----------
    v : float[3]
        The velocity vector (in m/s).

    Returns
    -------
    gamma : float
        The relativistic factor.
    '''

    return 1.0 / np.sqrt(1 - dot(v, v) / sp.c**2)


@njit
def local_onb(r, b_field, t=0.):
    '''
    Constructs an orthonormal basis at a given location with the z axis along the magnetic field and the x axis directed toward the origin.

    Parameters
    ----------
    r : float[3]
        The origin of the new coordinate system.

    b_field(r, t=0.) : function
        The magnetic field function (this is obtained through the currying functions in fields.py). Accepts a
        position (float[3]) and time (float). Returns the magnetic field vector (float[3]) at that point in spacetime.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    Returns
    -------
    local_x : float[3]
        Local x axis unit vector.

    local_y : float[3]
        Local y axis unit vector.

    local_z : float[3]
        Local z axis unit vector.
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
    ----------
    r : float[3]
        The location (in m) of the particle.

    K : float
        The kinetic energy (in eV) of the particle.

    m : float
        The mass (in kg) of the particle.

    b_field(r, t=0.) : function
        The magnetic field function (this is obtained through the currying functions in fields.py). Accepts a
        position (float[3]) and time (float). Returns the magnetic field vector (float[3]) at that point in spacetime.

    pitch_angle : float
        The angle the velocity vector makes with respect to the magnetic field (in radians).

    phase_angle : float
        The angle the velocity vector makes with respect to (r x B) x B (in radians).

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    Returns
    -------
    v : float[3]
        The velocity vector (in m/s) of the particle.
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
def grad(field, r, t=0., eps=1e-6):
    '''
    Numerically finds the gradient of a field at a given point.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    r : float[3]
        The location at which to find the gradient.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    eps : float, optional
        The size of epsilon for use in the definition of the partial derivative. Defaults to 1e-6.

    Returns
    -------
    grad : float[3]
        The gradient of the field at the given point.
    '''

    x_offset = np.array([eps, 0., 0.])
    y_offset = np.array([0., eps, 0.])
    z_offset = np.array([0, 0., eps])

    grad = np.zeros(3)
    grad[0] = (np.linalg.norm(field(r + x_offset, t)) - np.linalg.norm(field(r - x_offset, t))) / (2 * eps)
    grad[1] = (np.linalg.norm(field(r + y_offset, t)) - np.linalg.norm(field(r - y_offset, t))) / (2 * eps)
    grad[2] = (np.linalg.norm(field(r + z_offset, t)) - np.linalg.norm(field(r - z_offset, t))) / (2 * eps)

    return grad


@njit
def jacobian(field, r, t=0., eps=1e-6):
    '''
    Numerically finds the Jacobian of a field at a given point.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    r : float[3]
        The location at which to find the Jacobian.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    eps : float, optional
        The size of epsilon for use in the definition of the partial derivative. Defaults to 1e-6.

    Returns
    -------
    jac : float[3, 3]
        The Jacobian of the field at the given point.
    '''

    x_offset = np.array([eps, 0., 0.])
    y_offset = np.array([0., eps, 0.])
    z_offset = np.array([0, 0., eps])

    jac = np.zeros((3, 3))
    jac[:, 0] = (field(r + x_offset, t) - field(r - x_offset, t)) / (2 * eps)
    jac[:, 1] = (field(r + y_offset, t) - field(r - y_offset, t)) / (2 * eps)
    jac[:, 2] = (field(r + z_offset, t) - field(r - z_offset, t)) / (2 * eps)

    return jac


@njit
def flc(field, r, t=0., eps=1e-1):
    '''
    Numerically finds the radius of curvature of the field line at a given point.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    r : float[3]
        The location at which to find the field line curvature.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    eps : float, optional
        The size of epsilon for use in the definition of the partial derivative. Defaults to 1e-6.

    Returns
    -------
    Rc : float
        The radius of curvature (in m).
    '''

    x_offset = np.array([eps, 0.0, 0.0])
    y_offset = np.array([0.0, eps, 0.0])
    z_offset = np.array([0.0, 0.0, eps])
    
    b = field(r, t)
    b /= np.linalg.norm(b)

    fx1 = field(r + x_offset, t)
    fx1 /= np.linalg.norm(fx1)
    
    fx0 = field(r - x_offset, t)
    fx0 /= np.linalg.norm(fx0)
    
    fy1 = field(r + y_offset, t)
    fy1 /= np.linalg.norm(fy1)
    
    fy0 = field(r - y_offset, t)
    fy0 /= np.linalg.norm(fy0)
    
    fz1 = field(r + z_offset, t)
    fz1 /= np.linalg.norm(fz1)
    
    fz0 = field(r - z_offset, t)
    fz0 /= np.linalg.norm(fz0)

    J = np.zeros((3, 3))
    J[:, 0] = (fx1 - fx0) / (2 * eps)
    J[:, 1] = (fy1 - fy0) / (2 * eps)
    J[:, 2] = (fz1 - fz0) / (2 * eps)

    return (1.0 / np.linalg.norm(np.dot(J, b)))


@njit
def field_line(field, r, t=0., tol=1e-5, max_iter=1000, planar=False):
    '''
    Traces a field line of a given magnetosphere model using the Runge-Kutta-Fehlberg method. Use this for the Tsyganenko and IGRF models.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    r : float[3]
        A location which intersects the desired field line.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    tol : float, optional
        The tolerance of the RK45 solver. Defaults to 1e-5.

    max_iter : int, optional
        The maximum iterations to run the RK45 solver. Defaults to 1000.

    Returns
    -------
    rr : float[N, 3]
        An array of points marking the field line. Begins and stops where the field line is 0.5 Re away from the origin.
    '''

    def rk45_step(field, r, h, tol, direction):
        a = np.sign(direction)
        
        k1 = a * field(r, t)
        k1 /= np.linalg.norm(k1)
        k1 *= h

        k2 = a * field(r + 0.25 * k1, t)
        k2 /= np.linalg.norm(k2)
        k2 *= h

        k3 = a * field(r + 0.09375 * k1 + 0.28125 * k2, t)
        k3 /= np.linalg.norm(k3)
        k3 *= h

        k4 = a * field(r + 0.87938097405553 * k1 - 3.2771961766045 * k2 + 3.3208921256259 * k3, t)
        k4 /= np.linalg.norm(k4)
        k4 *= h

        k5 = a * field(r + 2.0324074074074 * k1 - 8 * k2 + 7.1734892787524 * k3 - 0.20589668615984 * k4, t)
        k5 /= np.linalg.norm(k5)
        k5 *= h

        k6 = a * field(r - 0.2962962962963 * k1 + 2 * k2 - 1.3816764132554 * k3 + 0.45297270955166 * k4 - 0.275 * k5, t)
        k6 /= np.linalg.norm(k6)
        k6 *= h

        y_plus_1 = r + 0.11574074074074 * k1 + 0.54892787524366 * k3 + 0.53533138401559 * k4 - 0.2 * k5
        z_plus_1 = r + 0.11851851851852 * k1 + 0.51898635477583 * k3 + 0.50613149034202 * k4 - 0.18 * k5 + 0.036363636363636 * k6

        t_plus_1 = z_plus_1 - y_plus_1
        mag_t_plus_1 = np.linalg.norm(t_plus_1)

        if mag_t_plus_1 == 0:
            h = 1.8 * h
            return z_plus_1, h
        
        h = 0.9 * h * min(max(np.sqrt(tol / (2 * mag_t_plus_1)), 0.3), 2)
        
        return z_plus_1, h

    rrb = np.zeros((1, 3))
    rrb[0] = r

    h = 1e4

    i = 0
    while True:
        r, h = rk45_step(field, r, h, tol, -1)

        if planar:
            if np.linalg.norm(np.array([r[0], r[2]])) <= 0.2 * Re or i > max_iter:
                break
        else:
            if np.linalg.norm(r) <= 0.2 * Re or i > max_iter:
                break

        k = np.zeros((1, 3))
        k[0] = r
        rrb = np.append(rrb, k, axis=0)

        i += 1
        
    r = np.copy(rrb[0])
    
    rrf = np.zeros((1, 3))
    rrf[0] = r
   
    i = 0
    while True:
        r, h = rk45_step(field, r, h, tol, 1)

        if planar:
            if np.linalg.norm(np.array([r[0], r[2]])) <= 0.2 * Re or i > max_iter:
                break
        else:
            if np.linalg.norm(r) <= 0.2 * Re or i > max_iter:
                break

        k = np.zeros((1, 3))
        k[0] = r
        rrf = np.append(rrf, k, axis=0)

        i += 1
        
    rr = np.append(rrf[::-1], rrb, axis=0)
    
    return rr


@njit(parallel=True)
def b_along_path(field, rr, t=0.):
    '''
    Gives the magnetic field along a path.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    rr : float[N, 3]
        An array of points marking the path.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    Returns
    -------
    field_vec : float[N, 3]
        The field vector at each point along the path.

    field_mag : float[N]
        The strength of the field along the path.

    field_rad_mag : float[N]
        The strength of the radial (x and y) field along the path.
    '''

    steps = len(rr[:, 0])
    field_vec = np.zeros((steps, 3))
    field_mag = np.zeros(steps)
    field_rad_mag = np.zeros(steps)
    
    for i in prange(steps):
        vec = field(rr[i], t)
        
        field_vec[i] = vec
        field_mag[i] = np.linalg.norm(vec)
        field_rad_mag[i] = np.sqrt(vec[0]**2 + vec[1]**2)
        
    return field_vec, field_mag, field_rad_mag


@njit(parallel=True)
def b_along_history(field, position, time):
    '''
    Gives the magnetic field along a history of positions.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    position : float[N, M, 3]
        An array of N particle paths consisting of M points each.

    time : float[M]
        An array of M timesteps shared with the position history.

    Returns
    -------
    b_along_history_v : float[N, M, 3]
        The field vector at each point along the history.
    '''

    num_particles = np.shape(position)[0]
    steps = np.shape(position)[1]

    b_along_history_v = np.zeros_like(position)
    
    for i in prange(num_particles):
        for j in prange(steps):
            if (position[i, j] == 0).all():
                continue
                
            b_along_history_v[i, j, :] = field(position[i, j, :], time[j])

    return b_along_history_v


@njit
def field_reversal(field, rr, t=0.):
    '''
    Finds the point along a path at which the radial magnitude of the field is at a minimum.
    For a field with no dipole tilt, this coincides with the location of maximum curvature along a field line.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    rr : float[N, 3]
        An array of points marking the path.

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    Returns
    -------
    r : float[3]
        The point of minimum radial field strength.
    '''

    b_vec, b_mag, b_rad_mag = b_along_path(field, rr, t)
    return rr[b_rad_mag.argmin()]


@njit(parallel=True)
def adiabaticity(field, rr, K, t=0., tol=1e-6, m=sp.m_e, q=-sp.e):
    '''
    For a given particle, returns the adiabaticity along a path. This is usually referred to by kappa = sqrt(R_c/rho),
    where R_c is the radius of curvature of the field line at that point and rho is the particle's gyroradius.

    Parameters
    ----------
    field(r, t=0.) : function
        The field function (this is obtained through the currying functions in fields.py). Accepts a position (float[3])
        and time (float). Returns the field vector (float[3]) at that point in spacetime.

    rr : float[N, 3]
        An array of points marking the path.

    K : float
        The kinetic energy of the particle (in eV).

    t : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to 0.

    m : float, optional
        The mass of the particle (in kg). Defaults to the electron mass.

    q : float, optional
        The electric charge of the particle (in C). Defaults to the electron charge (the negative unit charge).

    Returns
    -------
    kappa : float[N]
        The adiabaticity of each point along the path.
    '''

    hist_new = np.zeros((len(rr[:, 0])))

    gamma_v = 1 + eV_to_J(K) / (m * sp.c**2)
    v = (sp.c / gamma_v) * sqrt(gamma_v**2 - 1)

    for i in prange(len(rr[:, 0])):
        b = field(rr[i, :])

        rho_0 = gamma_v * m * v / (abs(q) * np.linalg.norm(b))
        
        R_c = flc(field, rr[i, :], eps=tol)
        hist_new[i] = sqrt(R_c / rho_0)

    return hist_new


@njit
def solve_traj(i, steps, dt, initial_conditions, particle_properties, integrator, drop_lost, downsample):
    '''
    Solves a single particle trajectory.

    Parameters
    ----------
    i : int
        The index of the particle to solve.

    steps : int
        The number of steps to iterate the solver.

    dt : float
        The duration of each step.

    velocity : float[N, M, 3]
        A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.

    initial_conditions : float[N, 4, 3]
        Array of initial conditions. The first index denotes the particle, the second the quantity (1 = position, 2 = velocity, 3 = magnetic field, 4 = electric field), and the third the dimension.

    particle_properties : float[N, 2]
        Array of particle properties. The first index denotes the mass and the second the charge.

    integrator(state, intrinsic, dt, step_num) : function
        The integrator function (this is obtained through the currying functions in integrators.py).

    drop_lost : bool
        Whether particles within 1 Re + 100 km of the origin should be dropped from the simulation.

    downsample : int
        Sample every N steps, with N = downsample.

    Returns
    -------
    hist_indiv : float[M, 4, 3]
        The complete history of a single particle.
    '''

    hist_indiv = np.zeros((steps, 4, 3))
    hist_indiv[0, :, :] = np.copy(initial_conditions[i, :, :])

    if (hist_indiv[0, :, :] == np.zeros((4, 3))).all():
        return np.zeros((int(steps // downsample), 4, 3))

    for j in range(steps - 1):
        hist_indiv[j + 1] = integrator(hist_indiv[j], particle_properties[i, :], dt, j)

        if drop_lost:
            if dot(hist_indiv[j + 1, 0, :], hist_indiv[j + 1, 0, :]) <= (Re + 100e3)**2:
                break

    return hist_indiv[::downsample, :, :]


@njit(parallel=True)
def solve_sys(history, initial_conditions, dt, particle_properties, integrator, drop_lost, downsample):
    num_particles = np.shape(history)[0]
    steps = np.shape(history)[1]

    for i in prange(num_particles):
        history[i, :] = solve_traj(i, steps, dt, initial_conditions[i], particle_properties[i], integrator, drop_lost, downsample)

    return history
