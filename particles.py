import numpy as np

from constants import *

class Particle:
    def __init__(self, r0, v_dir, E, q, m, integrator):
        self.q = q
        self.m = m

        gamma = (E * 1.602e-19) / (m * c**2) + 1
        v = c * np.sqrt(1 - gamma**(-2)) # m/s
        self.r = r0
        self.v = v_dir / np.linalg.norm(v_dir) * v

    def v_par(self, b_field):
        B = b_field.at(self.r)
        return np.dot(self.v, B) / np.linalg.norm(B)

    def v_perp(self, b_field):
        B = b_field.at(self.r)
        v_parallel = self.v_par(b_field)
        return self.v - v_parallel * B / np.linalg.norm(B)

    def p(self):
        return (self.m * self.v) / np.sqrt(1 - np.dot(self.v, self.v) / c**2)

    def E(self):
        return self.m * c**2 * (1 / np.sqrt(1 - np.dot(self.v, self.v) / c**2) - 1)

    def moment(self, b_field):
        B = b_field.at(self.r)
        v_perpendicular = self.v_perp(b_field)
        return self.m * 0.5 * np.dot(v_perpendicular, v_perpendicular) / np.linalg.norm(B)

    def pitch_angle(self, b_field):
        B = b_field.at(self.r)
        return np.arccos(np.dot(self.v, B) / (np.linalg.norm(self.v), np.linalg.norm(B)))

    def gyroradius(self, b_field):
        return self.m * self.v_perp(b_field) / ( abs(self.q) * np.linalg.norm(b_field.at(self.r)) )

    def gyrofreq(self, b_field):
        return abs(self.q) * np.linalg.norm(b_field.at(self.r)) / self.m
