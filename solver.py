import sys
import math as mt

import numpy as np
from scipy import signal

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import *
from particles import *
from integrators import *
from fields import *

class Solver:
    def __init__(self, e_field, b_field, integrator):
        self.e_field = e_field
        self.b_field = b_field
        self.integrator = integrator
        self.steps = None

        self.particles = []

    def add_particle(self, particle):
        self.particles.append(particle)

    def solve(self, T, diagnostic_list):
        self.steps = int(mt.ceil(T / self.integrator.dt))

        for particle in self.particles:
            for param in diagnostic_list:
                particle.history[param] = np.zeros((self.steps, diagnostics[param]['dims']))

            for t in range(self.steps):
                particle.r, particle.v = self.integrator.step(particle, self.e_field, self.b_field)

                for param in diagnostic_list:
                    if diagnostics[param]['requires_B']:
                        B = self.b_field.at(particle.r)
                        particle.history[param][t] = diagnostics[param]['func'](particle, B)
                    else:
                        particle.history[param][t] = diagnostics[param]['func'](particle)

                particle.equatorial_pitch_angle()

    def plot_3d(self, diagnostic_list):
        fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
        ax = fig.gca(projection='3d')

        for particle in self.particles:
            for param in diagnostic_list:
                plt.plot(particle.history[param][:, 0], particle.history[param][:, 1], particle.history[param][:, 2])
                ax.auto_scale_xyz([2*Re, 6*Re], [-2*Re, 2*Re], [-2*Re, 2*Re])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        plt.show()

    def plot_2d(self, diagnostic_list):
        for particle in self.particles:
            for i, param in enumerate(diagnostic_list):
                fig = plt.figure(i)
                plt.plot(np.arange(0, self.steps) * self.integrator.dt, particle.history[param][:])
            ax = fig.gca()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(diagnostics[param]['label'])
            plt.grid()

        plt.show()

if __name__ == '__main__':
    # Setup the electric and magnetic fields of the system. Superposition to be implemented in the future.
    e_field = UniformField(0, 'z')
    b_field = EarthDipole()

    # Create the system to be studied.
    T = 1e0
    dt = 1e-4
    system = Solver(e_field, b_field, BorisRel(dt))

    # Add particles. The first array is the particle's position, the second is the particle's velocity.
    # The third argument is the particle's energy (in eV), the fourth is its charge and the fifth is its mass.

    system.add_particle( Particle(np.array([5 * Re, 0., 0.]), np.array([0., 0.5, 0.5]), 50e5, -qe, me) )

    # Call these last. They solve the trajectories and plot the solutions
    system.solve(T, ['position', 'gyrofreq', 'par_velocity'])
    # system.plot_3d(['position'])
    system.plot_2d(['position', 'gyrofreq', 'par_velocity'])
