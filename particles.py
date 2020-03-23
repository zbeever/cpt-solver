import numpy as np

from constants import *

class Particle:
    def __init__(self, r0, v_dir, E, q, m):
        self.q = q
        self.m = m

        gamma = (E * 1.602e-19) / (m * c**2) + 1
        v = c * np.sqrt(1 - gamma**(-2)) # m/s

        self.r = r0
        self.v = v_dir / np.linalg.norm(v_dir) * v

        self.history = {}

    def r(self):
        return self.r

    def v(self):
        return self.v

    def v_par(self, B):
        return np.dot(self.v, B) / np.linalg.norm(B)

    def v_perp(self, B):
        v_parallel = self.v_par(B)
        v_perpendicular = self.v - v_parallel * B / np.linalg.norm(B)
        return np.linalg.norm(v_perpendicular)

    def p(self):
        return (self.m * self.v) / np.sqrt(1 - np.dot(self.v, self.v) / c**2)

    def E(self):
        return self.m * c**2 * (1 / np.sqrt(1 - np.dot(self.v, self.v) / c**2) - 1)

    def moment(self, B):
        v_perpendicular = self.v_perp(B)
        return self.m * 0.5 * v_perpendicular**2 / np.linalg.norm(B)

    def pitch_angle(self, B):
        return np.mod(np.arctan(self.v_perp(B) / self.v_par(B)), np.pi)

    def equatorial_pitch_angle(self):
        if 'pitch_angle' in self.history and 'position' in self.history:
            z = self.history['position'][:, 2]
            z_sign = np.sign(z)
            equatorial_crossing = ((np.roll(z_sign, 1) - z_sign) != 0).astype(int)
            indices = np.argwhere(equatorial_crossing == 1)

            eq_pitch_angles = []

            for i in indices:
                eq_pitch_angles.append((self.history['pitch_angle'][i] + self.history['pitch_angle'][i - 1]) * 0.5)

            self.history['equatorial_pitch_angle'] = eq_pitch_angles

    def gyroradius(self, B):
        return self.m * self.v_perp(B) / ( abs(self.q) * np.linalg.norm(B) )

    def gyrofreq(self, B):
        return abs(self.q) * np.linalg.norm(B) / self.m

diagnostics = {
    'position': {'func': Particle.r, 'requires_B': False, 'dims': 3, 'label': 'Position (m)'},
    'velocity': {'func': Particle.v, 'requires_B': False, 'dims': 3, 'label': 'Velocity (m/s)'},
    'perp_velocity': {'func': Particle.v_par, 'requires_B': True, 'dims': 1, 'label': 'Velocity (m/s)'},
    'par_velocity': {'func': Particle.v_perp, 'requires_B': True, 'dims': 1, 'label': 'Velocity (m/s)'},
    'momentum': {'func': Particle.p, 'requires_B': False, 'dims': 3, 'label': 'Momentum (kgm/s)'},
    'energy': {'func': Particle.E, 'requires_B': False, 'dims': 1, 'label': 'Energy (eV)'},
    'pitch_angle': {'func': Particle.pitch_angle, 'requires_B': True, 'dims': 1, 'label': 'Pitch Angle (radians)'},
    'gyroradius': {'func': Particle.gyroradius, 'requires_B': True, 'dims': 1, 'label': 'Gyroradius (m)'},
    'gyrofreq': {'func': Particle.gyrofreq, 'requires_B': True, 'dims': 1, 'label': 'Gyrofrequency (rad/s)'}
}
