import numpy as np

import sys
import math as mt

from constants import *
from particles import *
from fields import *
from integrators import *
from distributions import *

import numba
from numba import float32, jit, njit

class System:
    def __init__(self, num_particles, T, dt, integrator = BorisRel(), b_field = ZeroField(), e_field = ZeroField()):
        self.T = T
        self.dt = dt

        self.steps = int(mt.ceil(T / dt))
        self.num_particles = num_particles

        # [particle index, time index, property, dimension]
        # Property: 0 = position, 1 = velocity, 2 = B field, 3 = E field
        # Dimension: 0 = x, 1 = y, 2 = z
        self.history = np.zeros((num_particles, self.steps, 4, 3))

        # [mass index, charge index]
        self.intrinsic = np.zeros((num_particles, 2))

        self.integrator = integrator
        self.b_field = b_field
        self.e_field = e_field

    def ONB(self, r, t = 0.0):
        B = self.b_field.at(r, t)

        local_z = B
        if np.dot(local_z, local_z) == 0:
            local_z = np.array([0., 0., 1.])
        else:
            local_z = local_z / np.linalg.norm(local_z)

        local_x = -np.asarray(r)
        local_x = local_x - np.dot(local_x, B) * B / np.dot(B, B)
        if np.dot(local_x, local_x) == 0:
            local_x = np.array([-1., 0., 0.])
        else:
            local_x = local_x / np.linalg.norm(local_x)

        local_y = np.cross(local_z, local_x)
        return local_x, local_y, local_z

    def get_history(self):
        return self.history

    def velocity_vec(self, r, K, m, pitch_angle, phase_angle, t = 0.0):
        local_x, local_y, local_z = self.ONB(r, t)

        v_dir = np.sin(pitch_angle) * np.cos(phase_angle) * local_x + np.sin(pitch_angle) * np.sin(phase_angle) * local_y + np.cos(pitch_angle) * local_z

        gamma_v = eV_to_J(K) / (m * c ** 2.0) + 1.0
        v_mag = c * mt.sqrt(1. - gamma_v ** (-2.0))

        if np.dot(v_dir, v_dir) == 0.0:
            return np.array([0.0, 0.0, 0.0])
        else:
            return v_dir / np.linalg.norm(v_dir) * v_mag

    def populate(self, pos_dist, E_dist, pitch_angle_dist, phase_angle_dist, m_dist = Delta(me), q_dist = Delta(-qe), t = 0.0):
        for i in range(self.num_particles):
            r = pos_dist.sample()
            K = E_dist.sample()
            m = m_dist.sample()
            q = q_dist.sample()

            pitch_angle = pitch_angle_dist.sample()
            phase_angle = phase_angle_dist.sample()

            self.history[i, 0, 0] = r
            self.history[i, 0, 1] = self.velocity_vec(r, K, m, pitch_angle, phase_angle)
            self.history[i, 0, 2] = self.b_field.at(r, t)
            self.history[i, 0, 3] = self.e_field.at(r, t)
            self.intrinsic[i, 0] = m
            self.intrinsic[i, 1] = q

    def solve(self):
        for i in range(self.num_particles):
            for j in range(self.steps - 1):
                self.history[i, j + 1, :] = self.integrator.step(self.history[i, j], self.intrinsic[i], self.e_field, self.b_field, j, self.dt)
