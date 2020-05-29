import numpy as np
from matplotlib import pyplot as plt
from constants import *

class Field:
    def __init__(self):
        return

    def at(self, r, t):
        return

    def __add__(self, other):
        return CombinedField(self, other)

class CombinedField(Field):
    def __init__(self, field0, field1):
        self.field0 = field0
        self.field1 = field1

    def at(self, r):
        return self.field0.at(r) + self.field1.at(r)

class ZeroField(Field):
    # A uniform field. Simply specify the strength and the axis it
    # should be parallel to, 'x', 'y', or 'z'

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

def plot_field(field, x_len, y_len, z_len, nodes):
    # Given one of the above fields, the dimensions of the area to plot, and the nodes
    # at which to display the field values, plots the field.

    x = np.linspace(-x_len * 0.5, x_len * 0.5, nodes)
    y = np.linspace(-y_len * 0.5, y_len * 0.5, nodes)
    z = np.linspace(-z_len * 0.5, z_len * 0.5, nodes)

    xv, yv, zv = np.meshgrid(x, y, z)
    uv, vv, wv = np.meshgrid(x, y, z)

    for i in range(nodes):
        for j in range(nodes):
            for k in range(nodes):
                uv[i][j][k], vv[i][j][k], wv[i][j][k] = field.at([xv[i][j][k], yv[i][j][k], zv[i][j][k]])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(xv, yv, zv, uv, vv, wv, length=0.5, normalize=True)
    plt.show()
