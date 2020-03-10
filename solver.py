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

class System:
    def __init__(self, e_field, b_field, integrator):
        self.e_field = e_field
        self.b_field = b_field
        self.integrator = integrator

        self.particles = []

    def add_particle(self, particle):
        self.particles.append(particle)

    def plot(self, T):
        steps = int(mt.ceil(T / self.integrator.dt))
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.gca(projection='3d')

        for particle in self.particles:
            R = np.zeros((steps, 3))
            V = np.zeros((steps, 3))
            mu = np.zeros((steps, 1))
            B = np.zeros((steps, 1))
            wn = np.zeros((steps, 1))

            for t in range(steps):
                r, v = self.integrator.step(particle, self.e_field, self.b_field)
                particle.r, particle.v = r, v

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

if __name__ == '__main__':
    # Setup the electric and magnetic fields of the system. Superposition to be implemented in the future.
    e_field = UniformField(0, 'z')
    b_field = EarthDipole()

    # Create the system to be studied.
    T = 1
    dt = 1e-4
    system = System(e_field, b_field, BorisRel(dt))

    # Add particles. The first array is the particle's position, the second is the particle's velocity.
    # The third argument is the particle's energy (in eV), the fourth is its charge and the fifth is its mass.

    system.add_particle( Particle(np.array([5 * Re, 0., 0.]), np.array([0., 0.5, 1.]), 50e6, -qe, me, BorisRel(dt)) )

    # Call these last. They solve the trajectories and plot the solutions
    system.plot(T)
