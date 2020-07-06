from distributions import *
from utils import *
from fields import *
from field_utils import *
from integrators import *
from scipy import constants as sp
from constants import *
import numpy as np
from math import sqrt
from numba import njit


def solver(integrator):
    @njit
    def solve(history, intrinsic, dt):
        num_particles = np.shape(intrinsic)[0]
        steps = np.shape(history)[1]

        for i in range(num_particles):
            for j in range(steps - 1):
                history[i, j + 1] = integrator(history[i, j], intrinsic[i], dt, j)

    return solve


def populate(num_particles, steps, e_field, b_field, pos_dist, E_dist, pitch_angle_dist, phase_angle_dist, m_dist = delta(sp.m_e), q_dist = delta(-sp.e), t = 0.):
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


def populate_by_eq_pa(num_particles, steps, e_field, b_field, re_over_Re, E_dist, eq_pitch_angle_dist, phase_angle_dist, m_dist = delta(sp.m_e), q_dist = delta(-sp.e), t = 0.):
    history = np.zeros((num_particles, steps, 4, 3))
    intrinsic = np.zeros((num_particles, 2))

    for i in range(num_particles):
        mag_eq = np.array([-re_over_Re() * Re, 0, 0])
        rr = field_line(b_field, mag_eq, 1e1)

        K = E_dist()
        m = m_dist()
        q = q_dist()

        phase_angle = phase_angle_dist()

        r, pitch_angle = param_by_eq_pa(b_field, rr, eq_pitch_angle_dist())
        b = b_field(r)
        v = velocity_vec(r, K, m, b_field, pitch_angle, phase_angle)

        v_par = dot(b, v) / dot(b, b) * b
        v_perp = v - v_par
        gyrorad = gamma(v) * m * np.linalg.norm(v_perp) / (abs(q) * b)

        history[i, 0, 0] = r
        history[i, 0, 1] = v 
        history[i, 0, 2] = b_field(r, t)
        history[i, 0, 3] = e_field(r, t)
        intrinsic[i, 0] = m
        intrinsic[i, 1] = q

    return history, intrinsic
