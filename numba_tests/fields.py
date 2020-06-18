import numpy as np
import scipy.constants as sp
from geopack_numba import geopack as gp
from geopack_numba import t89 as test_t89
from numba import njit, jit
from matplotlib import pyplot as plt

from constants import *

def zero_field():
    @njit
    def field(r, t = 0.):
        return np.zeros(3)

    return field


def uniform_field(strength, axis):
    norm_axis = axis / np.linalg.norm(axis)
    field_vec = strength * norm_axis

    @njit
    def field(r, t = 0.):
        return field_vec

    return field


def harris_cs_model(b0, bn, d):
    @njit
    def field(r, t = 0.):
        return np.array([self.b0 * np.tanh(r[2] / self.d), 0., self.bn])

    return field


def magnetic_dipole(current, signed_area):
    m = current * signed_area

    @njit
    def field(r, t = 0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = sp.mu_0 / (4 * np.pi)

        return k * (3 * np.dot(m, r_unit) * r_unit - m) * r_mag**(-3)

    return field


def electric_dipole(charge, displacement):
    p = charge * displacement

    @njit
    def field(r, t = 0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = 1 / (4 * np.pi * sp.epsilon_0)

        return k * (3 * np.dot(p, r_unit) * r_unit - p) * r_mag**(-3)

    return field


def earth_dipole(t = 0.):
    gp.recalc(t)

    @njit
    def field(r, t = 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.dip(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def igrf(t = 0.):
    gp.recalc(t)

    @njit
    def field(r, t = 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def t89(Kp, t = 4.01e7, sw_v = np.array([-400., 0., 0.])):
    ps = gp.recalc(t, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t = 0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re
        
        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm))
        field_ext = np.asarray(test_t89.t89(Kp, ps, x_gsm, y_gsm, z_gsm))
        
        return (field_int + field_ext) * 1e-9

    return field


def t96(par, t0 = 4.01e7):
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
