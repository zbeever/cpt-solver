import numpy as np
import scipy.constants as sp

from fields import *

from numba import njit

def relativistic_boris(e_field, b_field):
    '''Relativistic Boris integrator described in DOI: 10.3847/1538-4365/aab114. Advances a particle over one timestep.
    This integrator has the advantage of preserving volume and, in the absence of electric fields, energy

    Parameters
    ==========
    e_field(r, t): The electric field function (this is obtained through the currying functions in fields.py).
    b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py).

    Returns
    =======
    step(state, intrinsic, dt, step_num): Function with particle state (4x1 numpy array), intrinsic properties (3x1 numpy array), time step (float) and step number (int).
    '''

    @njit
    def step(state, intrinsic, dt, step_num):
        # Used to calculate the current time in the case of time-varying fields.
        time = step_num * dt

        r = state[0]
        v = state[1]
        m = intrinsic[0]
        q = intrinsic[1]

        # The standard relativistic factor, gamma
        gamma_n = (1 - np.dot(v, v) / sp.c ** 2) ** -0.5

        # Spatial component of the four velocity
        u_n = gamma_n * v

        # Particle movement over first half of the timestep (using the initial velocity)
        x_n12 = r + v * 0.5 * dt

        # Field at this new location
        E = e_field(x_n12, time + 0.5 * dt)
        B = b_field(x_n12, time + 0.5 * dt)

        # Inversion of Lorentz equation to obtain the velocity
        u_minus = u_n + q * dt * 0.5 * E / m
        gamma_n12 = (1 + np.dot(u_minus, u_minus) / sp.c ** 2) ** 0.5
        t = B * q * dt * 0.5 / (m * gamma_n12)
        s = 2 * t / (1 + np.dot(t, t))
        u_plus = u_minus + np.cross(u_minus + np.cross(u_minus, t), s)
        u_n1 = u_plus + q * dt * 0.5 * E / m
        v_avg = (u_n1 + u_n) * 0.5 / gamma_n12

        # Particle movement over second half of the timestep (using this new velocity)
        u_n1 = u_n + (q / m) * (E + np.cross(v_avg, B)) * dt
        gamma_n1 = (1 + np.dot(u_n1, u_n1) / sp.c ** 2) ** 0.5
        x_n1 = x_n12 + u_n1 * 0.5 * dt / gamma_n1

        state_new = np.zeros((4, 3))
        state_new[0] = x_n1
        state_new[1] = u_n1 / gamma_n1
        state_new[2] = b_field(x_n1, time + dt)
        state_new[3] = e_field(x_n1, time + dt)

        return state_new
    return step


def nonrelativistic_boris(e_field, b_field):
    '''Nonrelativistic Boris integrator described on the Particle-in-cell Wikipedia page. Advances a particle over one timestep.

    Parameters
    ==========
    e_field(r, t): The electric field function (this is obtained through the currying functions in fields.py).
    b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py).

    Returns
    =======
    step(state, intrinsic, dt, step_num): Function with particle state (4x1 numpy array), intrinsic properties (3x1 numpy array), time step (float) and step number (int).
    '''

    @njit
    def step(state, intrinsic, dt, step_num):
        time = step_num * dt

        r = state[0]
        v = state[1]
        m = intrinsic[0]
        q = intrinsic[1]

        E = e_field(r, time)
        B = b_field(r, time)

        q_prime = dt * q * 0.5 / m
        h = q_prime * B
        s = 2 * h / (1 + np.dot(h, h))
        u = v + q_prime * E
        u_prime = u + np.cross(u + np.cross(u, h), s)

        state_new = np.zeros((4, 3))
        state_new[1] = u_prime + q_prime * E
        state_new[0] = r + history_new[1] * dt
        state_new[2] = b_field(history_new[0], time + dt)
        state_new[3] = e_field(history_new[0], time + dt)

        return state_new
    return step
