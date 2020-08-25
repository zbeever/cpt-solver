import numpy as np
import scipy.constants as sp
from numba import njit

from math import sqrt, cos, sin, acos, asin

from cptsolver.utils import dot, cross


def relativistic_boris(e_field, b_field):
    '''
    Relativistic Boris integrator described in DOI: 10.3847/1538-4365/aab114. Advances a particle over one timestep.
    This integrator has the advantage of preserving volume and, in the absence of electric fields, energy.

    Parameters
    ----------
    e_field(r, t=0.) : function
        The electric field function (this is obtained through the currying functions in fields.py). Accepts a
        position (float[3]) and time (float). Returns the electric field vector (float[3]) at that point in spacetime.

    b_field(r, t=0.) : function
        The magnetic field function (this is obtained through the currying functions in fields.py). Accepts a
        position (float[3]) and time (float). Returns the magnetic field vector (float[3]) at that point in spacetime.

    Returns
    -------
    step(state, particle_properties, dt, step_num) : function
        The integrator step function. Accepts the particle state (float[4, 3]) , particle properties (float[2]),
        time step (float) and step number (int). Returns the new particle state (float[4, 3]).
    '''

    @njit
    def step(state, particle_properties, dt, step_num):
        # Used to calculate the current time in the case of time-varying fields.
        time = step_num * dt

        x_n = state[0]
        v_n = state[1]
        m = particle_properties[0]
        q = particle_properties[1]

        # The standard relativistic factor, gamma
        gamma_n = 1.0 / sqrt(1.0 - dot(v_n, v_n) / sp.c**2)

        # Spatial component of the four velocity
        u_n = gamma_n * v_n

        # Particle movement over first half of the timestep (using the initial velocity)
        x_n12 = x_n + u_n / (2.0 * gamma_n) * dt

        # Field at this new location
        E = e_field(x_n12, time + 0.5 * dt)
        B = b_field(x_n12, time + 0.5 * dt)

        # Inversion of Lorentz equation to obtain the velocity
        u_minus = u_n + q * dt * 0.5 * E / m
        gamma_n12 = (1 + dot(u_minus, u_minus) / sp.c ** 2) ** 0.5
        t = B * q * dt * 0.5 / (m * gamma_n12)
        s = 2 * t / (1 + dot(t, t))
        u_plus = u_minus + cross(u_minus + cross(u_minus, t), s)
        u_n1 = u_plus + q * dt * 0.5 * E / m
        v_avg = (u_n1 + u_n) * 0.5 / gamma_n12
        u_n1 = u_n + (q / m) * (E + cross(v_avg, B)) * dt
        gamma_n1 = sqrt(sp.c**2 + dot(u_n1, u_n1)) / sp.c

        # Particle movement over second half of the timestep (using this new velocity)
        x_n1 = x_n12 + u_n1 * 0.5 * dt / gamma_n1

        state_new = np.zeros((4, 3))
        state_new[0] = x_n1
        state_new[1] = u_n1 / gamma_n1
        state_new[2] = b_field(x_n1, time + dt)
        state_new[3] = e_field(x_n1, time + dt)

        return state_new
    return step


def vay(e_field, b_field):
    @njit
    def step(state, particle_properties, dt, step_num):
        # Used to calculate the current time in the case of time-varying fields.
        time = step_num * dt

        x_n = state[0]
        v_n = state[1]
        m = particle_properties[0]
        q = particle_properties[1]

        # The standard relativistic factor, gamma
        gamma_n = 1.0 / sqrt(1.0 - dot(v_n, v_n) / sp.c**2)

        # Spatial component of the four velocity
        u_n = gamma_n * v_n

        # Particle movement over first half of the timestep (using the initial velocity)
        x_n12 = x_n + u_n / (2.0 * gamma_n) * dt

        # Field at this new location
        E = e_field(x_n12, time + 0.5 * dt)
        B = b_field(x_n12, time + 0.5 * dt)

        # Inversion of Lorentz equation to obtain the velocity
        u_n12 = u_n + (q * dt) / (2.0 * m) * (E + cross(u_n / gamma_n, B))
        u_prime = u_n12 + E * (q * dt) / (2.0 * m)
        tau = B * (q * dt) / (2.0 * m)
        u_star = dot(u_prime, tau / sp.c)
        gamma_prime = sqrt(1.0 + dot(u_prime, u_prime) / sp.c**2)
        sigma = gamma_prime**2.0 - dot(tau, tau)
        gamma_n1 = sqrt((sigma + sqrt(sigma**2.0 + 4.0*(dot(tau, tau) + u_star**2.0))) / 2.0)
        t = tau / gamma_n1
        s = 1.0 / (1.0 + dot(t, t))
        u_n1 = s * (u_prime + dot(u_prime, t) * t + cross(u_prime, t))

        # Particle movement over second half of the timestep (using this new velocity)
        x_n1 = x_n12 + u_n1 * 0.5 * dt / gamma_n1

        state_new = np.zeros((4, 3))
        state_new[0] = x_n1
        state_new[1] = u_n1 / gamma_n1
        state_new[2] = b_field(x_n1, time + dt)
        state_new[3] = e_field(x_n1, time + dt)

        return state_new
    return step


def higueracary(e_field, b_field):
    @njit
    def step(state, particle_properties, dt, step_num):
        # Used to calculate the current time in the case of time-varying fields.
        time = step_num * dt

        x_n = state[0]
        v_n = state[1]
        m = particle_properties[0]
        q = particle_properties[1]

        # The standard relativistic factor, gamma
        gamma_n = 1.0 / sqrt(1.0 - dot(v_n, v_n) / sp.c**2)

        # Spatial component of the four velocity
        u_n = gamma_n * v_n

        # Particle movement over first half of the timestep (using the initial velocity)
        x_n12 = x_n + u_n / (2.0 * gamma_n) * dt

        # Field at this new location
        E = e_field(x_n12, time + 0.5 * dt)
        B = b_field(x_n12, time + 0.5 * dt)

        # Inversion of Lorentz equation to obtain the velocity
        u_minus = u_n + (q * dt) / (2.0 * m) * E
        gamma_minus = sqrt(1.0 + dot(u_minus, u_minus) / sp.c**2)
        tau = B * (q * dt) / (2.0 * m)
        u_star = dot(u_minus, tau / sp.c)
        sigma = gamma_minus**2 - dot(tau, tau)
        gamma_plus = sqrt((sigma + sqrt(sigma**2.0 + 4.0 * (dot(tau, tau) + u_star**2.0))) / 2.0)
        t = tau / gamma_plus
        s = 1.0 / (1.0 + dot(t, t))
        u_plus = s * (u_minus + dot(u_minus, t) * t + cross(u_minus, t))
        u_n1 = u_plus + (q * dt) / (2.0 * m) * E + cross(u_plus, t) # This is different from the paper, which erroneously uses u_minus in the cross term
        gamma_n1 = sqrt(sp.c**2 + dot(u_n1, u_n1)) / sp.c

        # Particle movement over second half of the timestep (using this new velocity)
        x_n1 = x_n12 + u_n1 * 0.5 * dt / gamma_n1

        state_new = np.zeros((4, 3))
        state_new[0] = x_n1
        state_new[1] = u_n1 / gamma_n1
        state_new[2] = b_field(x_n1, time + dt)
        state_new[3] = e_field(x_n1, time + dt)

        return state_new
    return step


def nonrelativistic_boris(e_field, b_field):
    '''
    Nonrelativistic Boris integrator described on the Particle-in-cell Wikipedia page. Advances a particle over one timestep.

    Parameters
    ----------
    e_field(r, t=0.) : function
        The electric field function (this is obtained through the currying functions in fields.py).

    b_field(r, t=0.) : function
        The magnetic field function (this is obtained through the currying functions in fields.py).

    Returns
    -------
    step(state, particle_properties, dt, step_num) : function
        The integrator step function. Accepts the particle state (float[4, 3]) , particle properties (float[2]), time step (float) and step number (int).
    '''

    @njit
    def step(state, particle_properties, dt, step_num):
        time = step_num * dt

        r = state[0]
        v = state[1]
        m = particle_properties[0]
        q = particle_properties[1]

        E = e_field(r, time)
        B = b_field(r, time)

        q_prime = dt * q * 0.5 / m
        h = q_prime * B
        s = 2 * h / (1 + dot(h, h))
        u = v + q_prime * E
        u_prime = u + cross(u + cross(u, h), s)

        state_new = np.zeros((4, 3))
        state_new[1] = u_prime + q_prime * E
        state_new[0] = r + history_new[1] * dt
        state_new[2] = b_field(state_new[0], time + dt)
        state_new[3] = e_field(state_new[0], time + dt)

        return state_new
    return step
