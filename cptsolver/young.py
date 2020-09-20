import numpy as np
from scipy import constants as sp

from cptsolver.utils import eV_to_J


def epsilon(E, q, m, b0x, sigma, L_cs):
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
    p = sqrt(K**2 / sp.c**2 + 2. * K * m)
    return p / (np.abs(q) * (sigma * b0x) * (sigma * L_cs))


def zeta_1(sigma):
    '''
    zeta_1 parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    sigma : float
        The sigma parameter of the associated Harris model. This is the ratio of b0z / b0x.

    Returns
    -------
    zeta_1 : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return 3. + 2. * sigma**2


def zeta_2(sigma):
    '''
    zeta_2 parameter in Young et al. (2002), DOI: 10.1029/2000JA000294

    Parameters
    ----------
    sigma : float
        The sigma parameter of the associated Harris model. This is the ratio of b0z / b0x.

    Returns
    -------
    zeta_2 : float
        See Young et al. (2002), DOI: 10.1029/2000JA000294
    '''

    return sigma


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

    alpha_eqs = np.linspace(0, np.pi / 2, 1000)
    quantity = np.sin(omega_A(eps) * alpha_eqs) * np.cos(alpha_eqs)**b_A(eps)
    return 1. / np.amax(quantity)


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

    return 1.0513540 + 0.13513581*eps - 0.50787555*eps**2


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

    quantity = 0
    for n, qn in enumerate(coefs):
        quantity += qn * eps**(-n)


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

    coefs = [1.0663037, -1.0944973, 0.016679378, -0.00049938987]
    return q(eps, coefs)


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

    coefs = [-0.35533865, 0.12800347, 0.0017113113]
    return q(eps, coefs)


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

    coefs = [0.23156321, 0.15561211, -0.0018604330]
    return q(eps, coefs)


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

    coefs = [-0.49667826, -0.0081941799, 0.0013621659]
    return q(eps, coefs)


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

    coefs = [2.1127103, -2.1339384, 0.068354519, -0.0033623678]
    return q(eps, coefs)


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

    coefs = [-0.95013179, 0.34678755, -2.1132763e-5]
    return q(eps, coefs)


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

    coefs = [1.2278829, 0.18332446, 0.0041818697]
    return q(eps, coefs)


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

    coefs = [-0.31583655, -0.026062916, 0.0029116064]
    return q(eps, coefs)


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

    coefs = [-0.51057275, 0.93651781, -0.031690658]
    return q(eps, coefs)


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

    coefs = [1.3295169, 0.45892579, -0.018710078]
    return q(eps, coefs)


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

    coefs = [-0.53875507, 0.69089153, 0.017921165]
    return q(eps, coefs)


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
