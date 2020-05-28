import numpy as np
from constants import *

class Integrator:
    def __init__(self, dt):
        self.dt = dt
        return

    def step(self, particle, e_field, b_field):
        return

# The relativistic Boris integrator, detailed in DOI: 10.3847/1538-4365/aab114 
# This integrator has the advantage of preserving volume and, in the absence of electric fields, energy
class BorisRel(Integrator):
    def __init__(self, dt):
        self.dt = dt
        return

    def step(self, particle, e_field, b_field):
        # The standard relativistic factor, gamma
        gamma_n = (1 - np.dot(particle.v, particle.v) / c**2)**(-0.5)

        # Spatial component of the four velocity
        u_n = gamma_n * particle.v

        # Particle movement over first half of the timestep (using the initial velocity)
        x_n12 = particle.r + particle.v * 0.5 * self.dt

        # Field at this new location
        E = e_field.at(x_n12)
        B = b_field.at(x_n12)

        # Inversion of Lorentz equation to obtain the velocity
        u_minus = u_n + particle.q * self.dt * 0.5 * E / particle.m
        gamma_n12 = (1 + np.dot(u_minus, u_minus) / c**2)**(0.5)
        t = B * particle.q * self.dt * 0.5 / (particle.m * gamma_n12)
        s = 2 * t / (1 + np.dot(t, t))
        u_plus = u_minus + np.cross(u_minus + np.cross(u_minus, t), s)
        u_n1 = u_plus + particle.q * self.dt * 0.5 * E / particle.m
        v_avg = (u_n1 + u_n) * 0.5 / gamma_n12

        # Particle movement over second half of the timestep (using this new velocity)
        u_n1 = u_n + (particle.q / particle.m) * (E + np.cross(v_avg, B)) * self.dt
        gamma_n1 = (1 + np.dot(u_n1, u_n1) / c**2)**(0.5)
        x_n1 = x_n12 + u_n1 * 0.5 * self.dt / gamma_n1

        v = u_n1 / gamma_n1
        r = x_n1

        return r, v

# The nonrelativistic Boris integrator, detailed in the Particle-in-cell Wikipedia page
class BorisNonrel(Integrator):
    def __init__(self, dt):
        self.dt = dt
        return

    def step(self, particle, e_field, b_field):
        E = e_field.at(particle.r)
        B = b_field.at(particle.r)

        q_prime = self.dt * particle.q * 0.5 / particle.m
        h = q_prime * B
        s = 2 * h / (1 + np.dot(h, h))
        u = particle.v + q_prime * E
        u_prime = u + np.cross(u + np.cross(u, h), s)

        v = u_prime + q_prime * E
        r = particle.r + v * self.dt

        return r, v