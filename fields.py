import numpy as np
from utils import *
import scipy.constants as sp
from geopack_numba import geopack as gp
from geopack_numba import t89 as test_t89
from numba import njit, jit
from matplotlib import pyplot as plt

from constants import *

def zero_field():
    """Zero field. Returns the zero vector.
    """

    @njit
    def field(r, t = 0.):
        return np.zeros(3)

    return field


def uniform_field(strength, axis):
    """Uniform field.

    Args:
    strength (float): The strength of the field in T or V/m.
    axis (numpy array): Direction along which the field lines point.
    """

    norm_axis = axis / np.linalg.norm(axis)
    field_vec = strength * norm_axis

    @njit
    def field(r, t = 0.):
        return field_vec

    return field


def harris_cs_model(b0, bn, d):
    """Harris current sheet model.

    Args:
    B0_ (float): Magnitude of the x component of the magnetic field in the asymptotic region.
    Bn_ (float): Magnitude of the z component of the magnetic field.
    d_ (float): Scale length of the field reversal region.
    """

    @njit
    def field(r, t = 0.):
        return np.array([self.b0 * np.tanh(r[2] / self.d), 0., self.bn])

    return field


def magnetic_dipole(current, signed_area):
    """Magnetic dipole field formed from a current loop.

    Args:
    current (float): The value of the current in amperes.
    signed_area (numpy array): Normal vector to the plane of the loop whose length is numerically equal to the loop's enclosed area in m^2.
    """

    m = current * signed_area

    @njit
    def field(r, t = 0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = sp.mu_0 / (4 * np.pi)

        return k * (3 * np.dot(m, r_unit) * r_unit - m) * r_mag**(-3)

    return field


def electric_dipole(charge, displacement):
    """Electric dipole field formed from two opposite charges.

    Args:
    charge (float): The magnitude of one of the charges in C.
    displacement (numpy array): The vector pointing from the negative charge to the positive one in m.
    """

    p = charge * displacement

    @njit
    def field(r, t = 0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = 1 / (4 * np.pi * sp.epsilon_0)

        return k * (3 * np.dot(p, r_unit) * r_unit - p) * r_mag**(-3)

    return field


def earth_dipole_axis_aligned():
    """Dipole model of Earth's magnetic field with the dipole moment oriented along the z axis.
    """

    M = -8e15

    @njit
    def field(r, t = 0.0):
        [x, y, z] = r
        R = np.sqrt(x**2 + y**2 + z**2)

        B_x = 3 * M * (x * z) / (R**5)
        B_y = 3 * M * (y * z) / (R**5)
        B_z = M * (3 * z**2 - R**2) / (R**5)

        return np.array([B_x, B_y, B_z])

    return field


def earth_dipole(t0 = 4.01172e7):
    gp.recalc(t0)

    @njit
    def field(r, t = 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.dip(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def igrf(t0 = 4.01172e7):
    gp.recalc(t0)

    @njit
    def field(r, t = 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def t89(Kp, t0 = 4.01172e7, sw_v = np.array([-400., 0., 0.])):
    """Model of Earth's magnetic field consisting of a superposition of the Tsyganenko 1989 model (DOI: 10.1016/0032-0633(89)90066-4) and the IGRF model.

    Args:
    Kp (int): A mapping to the Kp geomagnetic activity index. Acceptable values range from 1 to 7, mapping to values between 0 and 10, inclusive.
    """

    ps = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t = 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(test_t89.t89(Kp, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t96(par, t0 = 4.01172e7):
    """A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 1996 model (DOI: 10.1029/96JA02735) and the IGRF model.

    Args:
    par (array): A 10-element array containing the model parameters.

    par[0] (Pdyn): The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.
    par[1] (Dst): The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value less than -50 nT indicates high geomagnetic activity. 
    par[2] (ByIMF): The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[3] (BzIMF): The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[4-9]: Not used
    """

    @jit
    def field(r, t = 0., sw_v = np.array([-400., 0., 0.])):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        ps = gp.recalc(t + t0, sw_v[0], sw_v[1], sw_v[2])
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(gp.t96.t96(par, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t01(par, t0 = 4.01e7):
    """A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2001 model (DOI: 10.1029/2001JA000220) and the IGRF model.

    Args:
    par (array): A 10-element array containing the model parameters.

    par[0] (Pdyn): The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.
    par[1] (Dst): The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value less than -50 nT indicates high geomagnetic activity. 
    par[2] (ByIMF): The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[3] (BzIMF): The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[4] (G1): A parameter capturing the cross-tail current's dependence on the solar wind. Mathematically, it is defined as the average of V*h(B_perp)*sin^3(theta/2) where V is the solar wind speed, B_perp is the transverse IMF component, theta is the IMF's clock angle, and h is a function that behaves as B_perp^2 for common values of the IMF. A typical value is 6.
    par[5] (G2): A parameter capturing the earthward / tailward shift of the tail current, defined as the average of a*V*Bs, where V is the solar wind speed, Bs is the southward component of the IMF (|Bz| for Bz < 0, 0 otherwise) and a = 0.005. A typical value is 10.
    par[6-9]: Not used

    """

    @jit
    def field(r, t = 0., sw_v = np.array([-400., 0., 0.])):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        ps = gp.recalc(t + t0, sw_v[0], sw_v[1], sw_v[2])
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(gp.t01.t01(par, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t04(par, t0 = 4.01e7):
    """A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2004 model (DOI: 10.1029/2004JA010798) and the IGRF model.

    Args:
    par (array): A 10-element array containing the model parameters.

    par[0] (Pdyn): The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.
    par[1] (Dst): The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value less than -50 nT indicates high geomagnetic activity. 
    par[2] (ByIMF): The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[3] (BzIMF): The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.
    par[4] (Wt1): Driving parameter of the inner part of the tail field. Saturates at 0.71 +- 0.05. Peak estimate: 4 to 12.
    par[5] (Wt2): Driving parameter of the outer part of the tail field. Saturates at 0.39 +- 0.05. Peak estimate: 3 to 7.
    par[6] (Ws): Driving parameter of the axially symmetric part of the ring current. Saturates at 3.3 +- 0.5. Peak estimate: 4 to 15.
    par[7] (Wp): Driving parameter of the partial ring current field. Saturates at 75 +- 30. Peak estimate 10 to 50.
    par[8] (Wb1): Driving parameter of the principle mode of the Birkeland current. Saturates at 6.4 +- 1.0. Peak estimate 7 to 30.
    par[9] (Wb2): Driving parameter of the secondary mode of the Birkeland current. Saturates at 0.88 +- 0.06. Peak estimate 20 to 100.
    """

    @jit
    def field(self, r, t = 0.0, sw_v = np.array([-400., 0., 0.])):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        ps = gp.recalc(t + self.t0, sw_v[0], sw_v[1], sw_v[2])
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(gp.t04.t04(par, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field

def plot_field(field, axis, nodes, x_lims, y_lims, size = (10, 10), t = 0.0):
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
