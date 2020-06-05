import numpy as np
from geopack import geopack as gp
from matplotlib import pyplot as plt
from constants import *

class Field:
    def __init__(self):
        return

    def at(self, r):
        return

class ZeroField(Field):
    def __init__(self):
        return

    def at(self, r):
        return np.array([0., 0., 0.])

class UniformField(Field):
    # A uniform field. Simply specify the strength and the axis it
    # should be parallel to

    def __init__(self, strength, axis):
        self.field = (axis / np.linalg.norm(axis)) * strength

    def at(self, r):
        return self.field

class MagneticDipoleField(Field):
    # A dipole field generated from a current loop.

    def __init__(self, current, signed_area):
        self.m = current * signed_area

    def at(self, r):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = mu0 / (4 * np.pi)

        return k * (3 * np.dot(self.m, r_unit) * r_unit - self.m) * r_mag**(-3)

class ElectricDipoleField(Field):
    # A dipole field generated from two nearby charges.

    def __init__(self, charge, displacement):
        self.p = charge * displacement

    def at(self, r):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = 1 / (4 * np.pi * epsilon0)

        return k * (3 * np.dot(self.p, r_unit) * r_unit - self.p) * r_mag**(-3)

class EarthDipole(Field):
    # The dipole model of the Earth's magnetic field.

    def __init__(self):
        self.M = -8e15

    def at(self, r):
        [x, y, z] = r
        R = np.sqrt(x**2 + y**2 + z**2)

        B_x = 3 * self.M * (x * z) / (R**5)
        B_y = 3 * self.M * (y * z) / (R**5)
        B_z = self.M * (3 * z**2 - R**2) / (R**5)

        return np.array([B_x, B_y, B_z])

class Tsyganenko89(Field):
    def __init__(self, Kp_):
        self.Kp = Kp_
        
    def at(self, r, t = 4.01e7, sw_v = np.array([-400., 0., 0.])):
        k = 1. / Re
        x_gsm = r[0] * k
        y_gsm = r[1] * k
        z_gsm = r[2] * k
        
        ps = gp.recalc(t, sw_v[0], sw_v[1], sw_v[2])
        bx, by, bz = gp.igrf_gsm(x_gsm, y_gsm, z_gsm)
        dbx, dby, dbz = gp.t89.t89(self.Kp, ps, x_gsm, y_gsm, z_gsm)
        
        return np.array([bx + dbx, by + dby, bz + dbz]) * 1e-9

def plot_field(field, axis, nodes, x_lims, y_lims, size = (10, 10)):
    x = np.linspace(x_lims[0], x_lims[1], nodes)
    y = np.linspace(y_lims[0], y_lims[1], nodes)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=size)

    if axis_num[axis] == 0:
        for i in range(nodes):
            for j in range(nodes):
                W, U[i][j], V[i][j] = field.at(np.array([1e-20, X[i][j], Y[i][j]]))
                ax.set_xlabel('$y$')
                ax.set_ylabel('$z$')
    elif axis_num[axis] == 1:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], W, V[i][j] = field.at(np.array([X[i][j], 1e-20, Y[i][j]]))
                ax.set_xlabel('$x$')
                ax.set_ylabel('$z$')
    elif axis_num[axis] == 2:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], V[i][j], W  = field.at(np.array([X[i][j], Y[i][j], 1e-20]))
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')

    color = 2 * np.log(np.hypot(U, V))
    ax.streamplot(X, Y, U, V, color=color, linewidth=1, cmap=plt.cm.jet, density=2, arrowstyle='wedge', arrowsize=1.)

    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    plt.show()
