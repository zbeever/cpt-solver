import numpy as np
from utils import *
import scipy.constants as sp
from geopack_numba import geopack as gp
from geopack_numba import t89 as ext_t89
from numba import njit, jit
from matplotlib import pyplot as plt


def zero_field():
    '''
    Zero field. Returns the zero vector.

    Parameters
    ----------
    None

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    @njit
    def field(r, t=0.):
        return np.zeros(3)

    return field


def uniform_field(strength, axis):
    '''
    Uniform field. Returns the same value at every point in space.

    Parameters
    ----------
    strength (float): The strength of the field (in T or V/m).
    axis (3x1 numpy array): Direction along which the field lines point.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    norm_axis = axis / np.linalg.norm(axis)
    field_vec = strength * norm_axis

    @njit
    def field(r, t=0.):
        return field_vec

    return field


def harris_cs_model(b0x, sigma, L_cs):
    '''
    Harris current sheet model, with the current sheet in the x-y plane.

    Parameters
    ----------
    b0x (float): The minimum value of the field.
    sigma (float): A parameter marking the perturbation strength of the current sheet. b0x * sigma is the radius of curvature of the field in the x-y plane.
    L_cs (float): The current sheet thickness.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    @njit
    def field(r, t=0.):
        return np.array([b0x * np.tanh(r[2] / L_cs), 0., sigma * b0x])

    return field


def magnetic_dipole(current, signed_area):
    '''
    Magnetic dipole field formed from a current loop.

    Parameters
    ----------
    current (float): The value of the current in amperes.
    signed_area (3x1 numpy array): Normal vector to the plane of the loop whose length is numerically equal to the loop's enclosed area (in m^2).

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    # Magnetic moment
    m = current * signed_area

    @njit
    def field(r, t=0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = sp.mu_0 / (4 * np.pi)

        return k * (3 * np.dot(m, r_unit) * r_unit - m) * r_mag**(-3)

    return field


def electric_dipole(charge, displacement):
    '''
    Electric dipole field formed from two opposite charges.

    Parameters
    ----------
    charge (float): The magnitude of one of the charges in C.
    displacement (3x1 numpy array): The vector pointing from the negative charge to the positive one in m.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    # Dipole moment
    p = charge * displacement

    @njit
    def field(r, t=0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = 1 / (4 * np.pi * sp.epsilon_0)

        return k * (3 * np.dot(p, r_unit) * r_unit - p) * r_mag**(-3)

    return field


def earth_dipole_axis_aligned():
    '''
    Dipole model of Earth's magnetic field with the dipole moment oriented along the z axis.

    Parameters
    ----------
    None

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    # Earth's magnetic moment with constants absorbed.
    M = -8e15

    @njit
    def field(r, t=0.):
        [x, y, z] = r
        R = np.sqrt(x**2 + y**2 + z**2)

        B_x = 3 * M * (x * z) / (R**5)
        B_y = 3 * M * (y * z) / (R**5)
        B_z = M * (3 * z**2 - R**2) / (R**5)

        return np.array([B_x, B_y, B_z])

    return field


def earth_dipole(t0=4.01172e7):
    '''
    Geopack dipole model of Earth's magnetic field.

    Parameters
    ----------
    t0 (float): The universal time (in seconds). Defaults to a value where the dipole tilt is approximately 0.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    gp.recalc(t0)

    @njit
    def field(r, t= 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.dip(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def igrf(t0=4.01172e7):
    '''
    The IGRF model of Earth's magnetic field.

    Parameters
    ----------
    t0 (float): The universal time (in seconds). Defaults to a value where the dipole tilt is approximately 0.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    gp.recalc(t0)

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def t89(Kp, t0=4.0118e7, sw_v=np.array([-400., 0., 0.])):
    '''
    Model of Earth's magnetic field consisting of a superposition of the Tsyganenko 1989 model (DOI: 10.1016/0032-0633(89)90066-4) and the IGRF model.

    Parameters
    ----------
    Kp (int): A mapping to the Kp geomagnetic activity index. Acceptable values range from 1 to 7, mapping to values between 0 and 6+, inclusive.
    t0 (float): The universal time (in seconds). Defaults to a value where the dipole tilt is approximately 0.
    sw_v (3x1 numpy array): The solar wind velocity vector in GSE coordinates. Defaults to v = [-400, 0, 0]

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    ps = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(ext_t89.t89(Kp, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t96(par, t0=4.0118e7):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 1996 model (DOI: 10.1029/96JA02735) and the IGRF model.

    Parameters
    ----------
    par (10 list): A 10-element list containing the model parameters.
    par[0]={Pdyn} (float): The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.
    par[1]={Dst} (float): The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value less than -50 nT indicates high geomagnetic activity. 
    par[2]={ByIMF} (float): The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[3]={BzIMF} (float): The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[4-9]: Not used
    t0 (float): The universal time (in seconds). Defaults to a value where the dipole tilt is approximately 0.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    @jit
    def field(r, t=0., sw_v=np.array([-400., 0., 0.])):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        ps = gp.recalc(t + t0, sw_v[0], sw_v[1], sw_v[2])
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(gp.t96.t96(par, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t01(par, t0=4.0118e7):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2001 model (DOI: 10.1029/2001JA000220) and the IGRF model.

    Parameters
    ----------
    par (10 list): A 10-element list containing the model parameters.
    par[0]={Pdyn} (float): The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.
    par[1]={Dst} (float): The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value less than -50 nT indicates high geomagnetic activity. 
    par[2]={ByIMF} (float): The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[3]={BzIMF} (float): The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[4]={G1} (float): A parameter capturing the cross-tail current's dependence on the solar wind. Mathematically, it is defined as the average of V*h(B_perp)*sin^3(theta/2) where V is the solar wind speed, B_perp is the transverse IMF component, theta is the IMF's clock angle, and h is a function that behaves as B_perp^2 for common values of the IMF. A typical value is 6.
    par[5]={G2} (float): A parameter capturing the earthward / tailward shift of the tail current, defined as the average of a*V*Bs, where V is the solar wind speed, Bs is the southward component of the IMF (|Bz| for Bz < 0, 0 otherwise) and a = 0.005. A typical value is 10.
    par[6-9]: Not used
    t0 (float): The universal time (in seconds). Defaults to a value where the dipole tilt is approximately 0.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    @jit
    def field(r, t=0., sw_v=np.array([-400., 0., 0.])):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        ps = gp.recalc(t + t0, sw_v[0], sw_v[1], sw_v[2])
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(gp.t01.t01(par, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t04(par, t0=4.01e7):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2004 model (DOI: 10.1029/2004JA010798) and the IGRF model.

    Parameters
    ----------
    par (10 list): A 10-element list containing the model parameters.
    par[0]={Pdyn} (float): The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.
    par[1]={Dst} (float): The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value less than -50 nT indicates high geomagnetic activity. 
    par[2]={ByIMF} (float): The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[3]={BzIMF} (float): The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[4]={Wt1} (float): Driving parameter of the inner part of the tail field. Saturates at 0.71 +- 0.05. Peak estimate: 4 to 12.
    par[5]={Wt2} (float): Driving parameter of the outer part of the tail field. Saturates at 0.39 +- 0.05. Peak estimate: 3 to 7.
    par[6]={Ws} (float): Driving parameter of the axially symmetric part of the ring current. Saturates at 3.3 +- 0.5. Peak estimate: 4 to 15.
    par[7]={Wp} (float): Driving parameter of the partial ring current field. Saturates at 75 +- 30. Peak estimate 10 to 50.
    par[8]={Wb1} (float): Driving parameter of the principle mode of the Birkeland current. Saturates at 6.4 +- 1.0. Peak estimate 7 to 30.
    par[9]={Wb2} (float): Driving parameter of the secondary mode of the Birkeland current. Saturates at 0.88 +- 0.06. Peak estimate 20 to 100.
    t0 (float): The universal time (in seconds). Defaults to a value where the dipole tilt is approximately 0.

    Returns
    -------
    field(r, t=0.): Function with a position (numpy array) and time (float) argument that returns the field at that spacetime coordinate.
    '''

    @jit
    def field(self, r, t= 0., sw_v=np.array([-400., 0., 0.])):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        ps = gp.recalc(t + self.t0, sw_v[0], sw_v[1], sw_v[2])
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(gp.t04.t04(par, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field
