import sys
import math as mt

import numpy as np
from scipy import signal

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

axis_num = {'x' : 0,
            'y' : 1,
            'z' : 2}

relativistic = True

mu0 = 1.256e-6 # H/m
epsilon0 = 8.854e-12 # F/m
me = 9.109e-31 # kg
mp = 1.672e-27 # kg
qe = 1.602e-19 # C
c = 3e8 # m/s
Re = 6.371e6 # m

class System:
    def __init__(self, electric_field, magnetic_field):
        self.e_field = electric_field
        self.b_field = magnetic_field
        self.particles = []

    def addParticle(self, particle):
        self.particles.append(particle)

    def plot(self, T, dt):
        steps = int(mt.ceil(T / dt))
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.gca(projection='3d')

        for particle in self.particles:

            R = np.zeros((steps, 3))
            V = np.zeros((steps, 3))
            mu = np.zeros((steps, 1))
            B = np.zeros((steps, 1))
            wn = np.zeros((steps, 1))

            for t in range(steps):
                r, v = particle.advance(self.e_field, self.b_field, dt)

                mu[t] = particle.moment(self.b_field)
                B[t] = np.linalg.norm(self.b_field.at(r))
                wn[t] = (abs(particle.q) * B[t]) / particle.m

                R[t, :] = r
                V[t, :] = v

            plt.plot(R[:, 0], R[:, 1], R[:, 2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Temporary solution to scale the axes before plotting
        ax.auto_scale_xyz([2*Re, 6*Re], [-2*Re, 2*Re], [-2*Re, 2*Re])
        plt.show()

class Particle:
    def __init__(self, r0, v_dir, E, q, m):
        v = 0

        if relativistic is False:
            v = np.sqrt(2 * E * 1.602e-19 / m)
        else:
            gamma = (E * 1.602e-19) / (m * c**2) + 1
            v = c * np.sqrt(1 - gamma**(-2)) # m/s

        self.q = q
        self.m = m
        self.r = r0
        self.v = v_dir / np.linalg.norm(v_dir) * v

    def v_par(self, b_field):
        B = b_field.at(self.r)
        return np.dot(self.v, B) / np.linalg.norm(B)

    def v_perp(self, b_field):
        B = b_field.at(self.r)
        v_parallel = self.v_par(b_field)
        return self.v - v_parallel * B / np.linalg.norm(B)

    def moment(self, b_field):
        B = b_field.at(self.r)
        v_perpendicular = self.v_perp(b_field)
        return self.m * 0.5 * np.dot(v_perpendicular, v_perpendicular) / np.linalg.norm(B)

    def pitch_angle(self, b_field):
        B = b_field.at(self.r)
        return np.arccos(np.dot(self.v, B) / (np.linalg.norm(self.v), np.linalg.norm(B)))

    def advance(self, e_field, b_field, dt):
        if relativistic is False:
            # Non-relativistic Boris method
            E = e_field.at(self.r)
            B = b_field.at(self.r)

            q_prime = dt * self.q * 0.5 / self.m
            h = q_prime * B
            s = 2 * h / (1 + np.dot(h, h))
            u = self.v + q_prime * E
            u_prime = u + np.cross(u + np.cross(u, h), s)

            self.v = u_prime + q_prime * E
            self.r += self.v * dt
        else:
            # Relativistic Boris method
            gamma_n = (1 - np.dot(self.v, self.v) / c**2)**(-0.5)
            u_n = gamma_n * self.v
            x_n12 = self.r + self.v * 0.5 * dt

            u_minus = u_n + self.q * dt * 0.5 * e_field.at(x_n12) / self.m
            gamma_n12 = (1 + np.dot(u_minus, u_minus) / c**2)**(0.5)
            t = b_field.at(x_n12) * self.q * dt * 0.5 / (self.m * gamma_n12)
            s = 2 * t / (1 + np.dot(t, t))
            u_plus = u_minus + np.cross(u_minus + np.cross(u_minus, t), s)
            u_n1 = u_plus + self.q * dt * 0.5 * e_field.at(x_n12) / self.m
            v_avg = (u_n1 + u_n) * 0.5 / gamma_n12

            u_n1 = u_n + (self.q / self.m) * (e_field.at(x_n12) + np.cross(v_avg, b_field.at(x_n12))) * dt
            gamma_n1 = (1 + np.dot(u_n1, u_n1) / c**2)**(0.5)
            x_n1 = x_n12 + u_n1 * 0.5 * dt / gamma_n1

            self.v = u_n1 / gamma_n1
            self.r = x_n1

        return (self.r, self.v)

class Field:
    def __init__(self):
        return

    def at(self, r, t):
        return

class UniformField(Field):
    # A uniform field. Simply specify the strength and the axis it
    # should be parallel to, 'x', 'y', or 'z'

    def __init__(self, strength, axis):
        self.field = np.array([0, 0, 0])
        self.field[axis_num[axis]] = strength

    def at(self, r):
        return self.field

class DipoleField(Field):
    # A dipole field generated from two nearby charges. Relative to
    # the (electric) dipole moment, strength corresponds to the magnitude
    # of qd while axis refers to the direction of the dipole's axis of symmetry

    def __init__(self, d, q, axis):
        self.p = np.array([0, 0, 0])
        self.p[axis_num[axis]] = d * q

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

if __name__ == '__main__':
    # Setup the electric and magnetic fields of the system. Superposition to be implemented in the future.
    e_field = UniformField(0, 'z')
    b_field = EarthDipole()

    # Create the system to be studied.
    system = System(e_field, b_field)

    # Add particles. The first array is the particle's position, the second is the particle's velocity.
    # The third argument is the particle's energy (in eV), the fourth is its charge and the fifth is its mass.

    system.addParticle( Particle(np.array([5 * Re, 0., 0.]), np.array([0., 0.5, 1.]), 50e6, -qe, me) )

    # Call these last. They solve the trajectories and plot the solutions
    T = 1
    dt = 1e-4
    system.plot(T, dt)
