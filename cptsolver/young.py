import numpy as np
from scipy import constants as sp
from numba import njit, prange

from cptsolver.utils import eV_to_J, field_line, b_along_path, flc, Re


def zetas_general(b_field, L, step_size=10, steps=1e4):
    '''
    Returns the zeta parameters along a field line of a magnetosphere model.

    Parameters
    ----------
    field(r, t=0.) : function
        The magnetic field model.

    L : float
        The L-shell value.

    step_size : float, optional
        The size of each step along the field line (in m). Defaults to 10.

    steps : int, optional
        The number of steps to take along the field line before computing the second derivative. Defaults to 10,000.

    Returns
    ------
    zeta1 : float
        The zeta_1 parameter, defined as R_c * d2R_c/ds2

    zeta2 : float
        The zeta_2 parameter, defined as (R_c**2/B_0) * d2B/ds2
        
    '''

    rr = field_line(b_field, np.array([-L * Re, 0., 0.]))
    bv, bm, brm = b_along_path(b_field, rr)

    cs_ind = bm.argmin()

    r_eq = np.linalg.norm(rr[cs_ind])
    R_c = flc(b_field, rr[cs_ind], 0, 1)

    b_plus_h = 0
    b_s0 = bm[cs_ind]
    b_minus_h = 0

    h = 0

    r_plus_h = np.copy(rr[cs_ind])
    b_plus_h = b_field(r_plus_h)
    b_plus_h /= np.linalg.norm(b_plus_h)

    r_minus_h = np.copy(rr[cs_ind])
    b_minus_h = b_field(r_minus_h)
    b_minus_h /= np.linalg.norm(b_minus_h)

    for i in range(int(steps)):
        r_old = np.copy(r_plus_h)
        r_plus_h += b_plus_h * step_size
        
        h_add = np.linalg.norm(r_plus_h - r_old)
        
        b_plus_h = b_field(r_plus_h)
        b_plus_h /= np.linalg.norm(b_plus_h)
        
        r_old = np.copy(r_minus_h)
        r_minus_h -= b_minus_h * step_size
        
        h_add += np.linalg.norm(r_minus_h - r_old)
        
        b_minus_h = b_field(r_minus_h)
        b_minus_h /= np.linalg.norm(b_minus_h)
        
        h += h_add * 0.5
        
    R_c_plus_h = flc(b_field, r_plus_h, 0, 1)
    R_c_minus_h = flc(b_field, r_minus_h, 0, 1)

    d2_Rc_ds2 = (R_c_plus_h - 2 * R_c + R_c_minus_h) / h**2
    d2_B_ds2 = (np.linalg.norm(b_field(r_plus_h)) - 2 * b_s0 + np.linalg.norm(b_field(r_minus_h))) / h**2

    zeta1 = R_c * d2_Rc_ds2
    zeta2 = (R_c**2 / b_s0) * d2_B_ds2

    return zeta1, zeta2


def zetas_harris(sigma):
    '''
    Returns the zeta parameters in a given Harris model.

    Parameters
    ----------
    sigma : float
        The sigma parameter of the associated Harris model. This is the ratio of b0z / b0x.

    Returns
    -------
    zeta_1 : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294

    zeta_2 : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return 3. + 2. * sigma**2, sigma


def epsilon_general(E, q, m, R_c, B0):
    '''
    Epsilon parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    E : float
        Kinetic energy of particle (in eV).

    q : float
        Charge of particle (in C).

    m : float
        Mass of particle (in kg).

    R_c : float
        The radius of curvature (in m) at the magnetic equator.

    B0 : float
        The magnitude of the magnetic field (in T) at the magnetic equator.

    Returns
    -------
    epsilon : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    K = eV_to_J(E)
    p = np.sqrt(K**2 / sp.c**2 + 2. * K * m)
    return p / (np.abs(q) * R_c * B0)


def epsilon_harris(E, q, m, b0x, sigma, L_cs):
    '''
    Epsilon parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    E : float
        Kinetic energy of particle in eV.

    q : float
        Charge of particle in C.

    m : float
        Mass of particle in kg.

    b0x : float
        The b0x parameter (in T) of the associated Harris model. The minimum B magnitude is given by sigma * b0x.

    sigma : float
        The sigma parameter of the associated Harris model. This is the ratio of b0z / b0x.

    L_cs : float
        The L_cs parameter (in m) of the associated Harris model. This is the current sheet thickness.

    Returns
    -------
    epsilon : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    K = eV_to_J(E)
    p = np.sqrt(K**2 / sp.c**2 + 2. * K * m)
    return p / (np.abs(q) * (sigma * b0x) * (sigma * L_cs))


@njit
def A_max(eps, z1, z2):
    '''
    A_max parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    z1 : float
        The zeta_1 parameter, see above.

    z2 : float
        The zeta_2 parameter, see above.

    Returns
    -------
    A_max : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return np.exp(c_A(eps)) * (z1**a_1A(eps) * z2**a_2A(eps) + D_A(eps))


@njit
def F_max(eps, z1, z2):
    '''
    F_max parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    z1 : float
        The zeta_1 parameter, see above.

    z2 : float
        The zeta_2 parameter, see above.

    Returns
    -------
    F_max : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return np.exp(c_F(eps)) * (z1**a_1F(eps) * z2**a_2F(eps) + D_F(eps))


@njit
def A(eps, z1, z2, alpha_eq):
    '''
    A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    z1 : float
        The zeta_1 parameter, see above.

    z2 : float
        The zeta_2 parameter, see above.

    alpha_eq : float
        The equatorial pitch angle in radians.

    Returns
    -------
    A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return A_max(eps, z1, z2) * N(eps) * np.sin(omega_A(eps) * alpha_eq) * np.cos(alpha_eq)**b_A(eps)


@njit
def F(eps, z1, z2, alpha_eq):
    '''
    F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    z1 : float
        The zeta_1 parameter, see above.

    z2 : float
        The zeta_2 parameter, see above.

    alpha_eq : float
        The equatorial pitch angle in radians.

    Returns
    -------
    F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return F_max(eps, z1, z2) * np.cos(omega_F(eps) * alpha_eq) * np.cos(alpha_eq)**b_F(eps)


@njit
def N(eps):
    '''
    N parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    N : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    alpha_eqs = np.linspace(0, np.pi / 2, 2000)
    quantity = np.sin(omega_A(eps) * alpha_eqs) * np.cos(alpha_eqs)**b_A(eps)
    return 1. / np.amax(quantity)


@njit
def omega_A(eps):
    '''
    omega_A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    omega_A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return 1.0513540 + 0.13513581 * eps - 0.50787555 * eps**2


@njit
def q(eps, coefs):
    '''
    Helper function for the fitted parameters whose coefficients are listed in Table 2 in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    coefs : float[N]
        The list of qn coefficients (see Table 2).

    Returns
    -------
    quantity : float
        The final polynomial in epsilon**-1
    '''

    N = len(coefs)
    return np.sum(coefs * eps**(-np.arange(0, N)))


@njit
def c_A(eps):
    '''
    c_A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    c_A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([1.0663037, -1.0944973, 0.016679378, -0.00049938987])
    return q(eps, coefs)


@njit
def a_1A(eps):
    '''
    a_1A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    a_1A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([-0.35533865, 0.12800347, 0.0017113113])
    return q(eps, coefs)


@njit
def a_2A(eps):
    '''
    a_2A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    a_2A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([0.23156321, 0.15561211, -0.0018604330])
    return q(eps, coefs)


@njit
def D_A(eps):
    '''
    D_A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    D_A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([-0.49667826, -0.0081941799, 0.0013621659])
    return q(eps, coefs)


@njit
def c_F(eps):
    '''
    c_F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    c_F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([2.1127103, -2.1339384, 0.068354519, -0.0033623678])
    return q(eps, coefs)


@njit
def a_1F(eps):
    '''
    a_1F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    a_1F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([-0.95013179, 0.34678755, -2.1132763e-5])
    return q(eps, coefs)


@njit
def a_2F(eps):
    '''
    a_2F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    a_2F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([1.2278829, 0.18332446, 0.0041818697])
    return q(eps, coefs)


@njit
def D_F(eps):
    '''
    D_F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    D_F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([-0.31583655, -0.026062916, 0.0029116064])
    return q(eps, coefs)


@njit
def b_A(eps):
    '''
    b_A parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    b_A : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([-0.51057275, 0.93651781, -0.031690658])
    return q(eps, coefs)


@njit
def omega_F(eps):
    '''
    omega_F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    omega_F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([1.3295169, 0.45892579, -0.018710078])
    return q(eps, coefs)


@njit
def b_F(eps):
    '''
    b_F parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    Returns
    -------
    b_F : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    coefs = np.array([-0.53875507, 0.69089153, 0.017921165])
    return q(eps, coefs)


@njit
def D_aa(eps, z1, z2, alpha_eq, T):
    '''
    The bounce-averaged equatorial pitch angle diffusion coefficient in Young et al. (2008), DOI: 10.1029/2006JA012133

    Parameters
    ----------
    eps : float
        The epsilon parameter, see above.

    z1 : float
        The zeta_1 parameter, see above.

    z2 : float
        The zeta_2 parameter, see above.

    alpha_eq : float
        The equatorial pitch angle in radians.

    T(alpha_eq) : function
        Functon returning the normalized bounce time given an equatorial pitch angle (in radians).

    Returns
    -------
    D_aa : float
        The bounce-averaged equatorial pitch angle diffusion coefficient.
    '''

    num = A(eps, z1, z2, alpha_eq)**2
    denom = 2. * T(alpha_eq) * np.sin(alpha_eq)**2 * np.cos(alpha_eq)**2
    return num / denom


@njit
def T_dipole(alpha_eq):
    '''
    Normalized bounce time for a dipole field.

    Parameters
    ----------
    alpha_eq : float
        The equatorial pitch angle in radians.

    Returns
    -------
    T : float
        The normalized bounce time.
    '''

    return 1.380173 - 0.639693 * np.sin(alpha_eq)**0.75
