from distributions import *
from fields import *
from integrators import *
from scipy import constants
from constants import *
import numpy as np
from math import sqrt

from numba import njit, jit, float32, int32, prange

def eV_to_J(E):
    return constants.e * E

def solver(integrator):
    def solve(history, intrinsic, dt):
        num_particles = np.shape(intrinsic)[0]
        steps = np.shape(history)[1]

        for i in range(num_particles):
            for j in range(steps - 1):
                history[i, j + 1] = integrator(history[i, j], intrinsic[i], dt, j)

    return solve


def ONB(r, b_field, t = 0.):
    B = b_field(r, t)

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


def velocity_vec(r, K, m, b_field, pitch_angle, phase_angle, t = 0.):
    local_x, local_y, local_z = ONB(r, b_field, t)

    v_dir = np.sin(pitch_angle) * np.cos(phase_angle) * local_x + np.sin(pitch_angle) * np.sin(phase_angle) * local_y + np.cos(pitch_angle) * local_z

    gamma_v = eV_to_J(K) / (m * sp.c ** 2.0) + 1.0
    v_mag = sp.c * np.sqrt(1. - gamma_v ** (-2.0))

    if np.dot(v_dir, v_dir) == 0.0:
        return np.array([0.0, 0.0, 0.0])
    else:
        return v_dir / np.linalg.norm(v_dir) * v_mag


def populate(num_particles, steps, e_field, b_field, pos_dist, E_dist, pitch_angle_dist, phase_angle_dist, m_dist = delta(constants.m_e), q_dist = delta(-constants.e), t = 0.):
    history = np.zeros((num_particles, steps, 4, 3))
    intrinsic = np.zeros((num_particles, 2))

    for i in range(num_particles):
        r = pos_dist()
        K = E_dist()
        m = m_dist()
        q = q_dist()

        pitch_angle = pitch_angle_dist()
        phase_angle = phase_angle_dist()

        history[i, 0, 0] = r
        history[i, 0, 1] = velocity_vec(r, K, m, b_field, pitch_angle, phase_angle)
        history[i, 0, 2] = b_field(r, t)
        history[i, 0, 3] = e_field(r, t)
        intrinsic[i, 0] = m
        intrinsic[i, 1] = q

    return history, intrinsic
