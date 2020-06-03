import numpy as np
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

def plot_field(field, axis, nodes, plot_size):
    x = np.linspace(-plot_size, plot_size, nodes)
    y = np.linspace(-plot_size, plot_size, nodes)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)

    if axis_num[axis] == 0:
        for i in range(nodes):
            for j in range(nodes):
                W, U[i][j], V[i][j] = field.at(np.array([1e-20, X[i][j], Y[i][j]]))
    elif axis_num[axis] == 1:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], W, V[i][j] = field.at(np.array([X[i][j], 1e-20, Y[i][j]]))
    elif axis_num[axis] == 2:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], V[i][j], W  = field.at(np.array([X[i][j], Y[i][j], 1e-20]))

    fig, ax = plt.subplots()
    color = 2 * np.log(np.hypot(U, V))
    ax.streamplot(X, Y, U, V, color=color, linewidth=1, cmap=plt.cm.jet, density=2, arrowstyle='wedge', arrowsize=1.)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-plot_size, plot_size)
    ax.set_ylim(-plot_size, plot_size)
    ax.set_aspect('equal')
    plt.show()
