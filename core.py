import sys
import math as mt
import numpy as np
from multiprocessing import Pool, freeze_support, cpu_count
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import *
from particles import *
from integrators import *
from fields import *

class Solver:
    def __init__(self, integrator, b_field = ZeroField(), e_field = ZeroField()):
        self.integrator = integrator
        self.b_field = b_field
        self.e_field = e_field
        self.steps = None
        self.particles = []
        self.diagnostics = []

    def add_particle(self, particle):
        self.particles.append(particle)

    def populate(self, num_particles, pos_dist, E_dist, pitch_angle_dist, phase_angle_dist):
        for i in range(num_particles):
            r = pos_dist.sample()
            K = E_dist.sample()
            pitch_angle = pitch_angle_dist.sample()
            phase_angle = phase_angle_dist.sample()

            B = self.b_field.at(r, 0) 
            
            local_z = B
            if np.dot(local_z, local_z) == 0:
                local_z = np.array([0., 0., 1.])
            else:
                local_z = local_z / np.linalg.norm(local_z)

            local_x = -r
            local_x = local_x - np.dot(local_x, B) * B / np.dot(B, B)
            if np.dot(local_x, local_x) == 0:
                local_x = np.array([-1., 0., 0.])
            else:
                local_x = local_x / np.linalg.norm(local_x)

            local_y = np.cross(local_z, local_x)

            v_dir = np.sin(pitch_angle) * np.cos(phase_angle) * local_x + np.sin(pitch_angle) * np.sin(phase_angle) * local_y + np.cos(pitch_angle) * local_z

            self.add_particle(Particle(r, v_dir, K, -qe, me))

    def solve(self, T, diagnostic_list):
        self.steps = int(mt.ceil(T / self.integrator.dt))
        self.diagnostics = diagnostic_list

        pool = Pool(cpu_count())
        histories = pool.map(self.solve_particle, self.particles)

        for i, particle in enumerate(self.particles):
            particle.history = histories[i]

    def solve_particle(self, particle):
        history = {}
        for param in self.diagnostics:
            if param in diagnostics:
                history[param] = np.zeros((self.steps, diagnostics[param]['dims']))

        for t in range(self.steps):
            particle.r, particle.v = self.integrator.step(particle, self.e_field, self.b_field, t)

            for param in self.diagnostics:
                if param in diagnostics:
                    if diagnostics[param]['requires_B']:
                        B = self.b_field.at(particle.r, t)
                        history[param][t] = diagnostics[param]['func'](particle, B)
                    else:
                        history[param][t] = diagnostics[param]['func'](particle)

        if 'eq_pitch_angle' in self.diagnostics:
            history['eq_pitch_angle'] = equatorial_pitch_angle(history['pitch_angle'], history['position'])

        if 'gca' in self.diagnostics:
            history['gca'] = gca(self.integrator.dt, history['position'], history['gyrofreq'])

        return history

    def plot_3d(self, diagnostic_list):
        fig = plt.figure(figsize=plt.figaspect(0.5) * 0.8)
        ax = fig.gca(projection='3d')

        for particle in self.particles:
            for param in diagnostic_list:
                plt.plot(particle.history[param][:, 0], particle.history[param][:, 1], particle.history[param][:, 2])
                # ax.auto_scale_xyz([np.amin(particle.history[param][:, 0]), np.amax(particle.history[param][:, 0])],
                #                   [np.amin(particle.history[param][:, 1]), np.amax(particle.history[param][:, 1])],
                #                   [np.amin(particle.history[param][:, 2]), np.amax(particle.history[param][:, 2])])
                # ax.auto_scale_xyz([-6*Re, 6*Re], [-6*Re, 6*Re], [-2*Re, 2*Re])

        # ax.dist = 12
        ax.set_xlabel('X (m)') #, labelpad=10)
        ax.set_ylabel('Y (m)') #, labelpad=7)
        # ax.set_yticklabels(['', -4, '', -2, '', 0, '', 2, '', 4])
        ax.set_zlabel('Z (m)') #, labelpad=3)

    def plot_2d(self, diagnostic_list):
        for particle in self.particles:
            for i, param in enumerate(diagnostic_list):
                fig = plt.figure(i)
                plt.plot(np.arange(0, self.steps) * self.integrator.dt, particle.history[param][:])
                ax = fig.gca()
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(diagnostics[param]['label'])
                plt.legend()
                plt.grid()

        plt.show()

if __name__ == '__main__':
    # Setup the fields of the system. Superposition to be implemented in the future.
    b_field = UniformField(2e3, 'z') # EarthDipole()

    # Create the system to be studied.
    T = 1e-1
    dt = 1e-4
    system = Solver(BorisRel(dt), b_field)

    # Add particles. The first array is the particle's position, the second is the direction of the particle's velocity vector,
    # the third argument is the particle's energy (in eV), the fourth is its charge (in C), and the fifth is its mass (in kg).

    # test_particle = Particle(np.array([5 * Re, 0., 0.]), np.array([0., 0.5, 0.5]), 50e6, -qe, me)
    test_particle = Particle(np.array([0., 0., 0.]), np.array([0., 0.5, 0.5]), 1e-5, -qe, me)
    system.add_particle(test_particle)

    # Call these last. They solve the trajectories and plot the solutions
    system.solve(T, ['position'])
    system.plot_2d(['position'])
    plt.show()
