import numpy as np
from scipy import signal

from constants import *
import logging as log

class Particle:
    def __init__(self, r0, v_dir, E, q, m):
        self.q = q
        self.m = m

        gamma = (E * 1.602e-19) / (m * c**2.) + 1.
        v = c * np.sqrt(1. - gamma**(-2.)) # m/s

        self.r = r0
        if (np.dot(v_dir, v_dir) == 0.):
            self.v = np.array([0., 0., 0.])
        else:
            self.v = v_dir / np.linalg.norm(v_dir) * v

        self.history = {}

    def r(self):
        return self.r

    def v(self):
        return self.v

    def v_par(self, B):
        v_dot_B = np.dot(self.v, B)
        B_norm = np.linalg.norm(B)

        return v_dot_B / B_norm

    def v_perp(self, B):
        v_parallel = self.v_par(B)
        v_perpendicular = self.v - v_parallel * B / np.linalg.norm(B)
        return np.linalg.norm(v_perpendicular)

    def p(self):
        gamma_v = gamma(np.linalg.norm(self.v))
        return gamma_v * self.m * self.v

    def E(self):
        E0 = self.m * c**2.
        gamma_v = gamma(np.linalg.norm(self.v))
        return E0 * (gamma_v - 1.)

    def moment(self, B):
        v_perpendicular = self.v_perp(B) # This is not exact; drift velocities should first be subtracted from v before computing the perpendicular velocity
        return .5 * self.m * v_perpendicular**2 / np.linalg.norm(B)

    def pitch_angle(self, B):
        return np.mod(np.arctan2(self.v_perp(B), abs(self.v_par(B))), np.pi)

    def gyroradius(self, B):
        gamma_v = gamma(np.linalg.norm(self.v))
        v_perp = self.v_perp(B)
        q_abs = abs(self.q)
        B_norm = np.linalg.norm(B)
        return gamma_v * self.m * v_perp / (q_abs * B_norm)

    def gyrofreq(self, B):
        q_abs = abs(self.q)
        B_norm = np.linalg.norm(B)
        gamma_v = gamma(np.linalg.norm(self.v))
        return q_abs * B_norm / (gamma_v * self.m)

    def b_strength(self, B):
        return np.linalg.norm(B)

def equatorial_pitch_angle(pitch_angle, position):
    z = position[:, 2]
    z_sign = np.sign(z)
    equatorial_crossing = ((np.roll(z_sign, 1) - z_sign) != 0).astype(int) # We assume the equator is located in the x-y plane at z = 0
    indices = np.argwhere(equatorial_crossing == 1)

    eq_pitch_angles = []

    for i in indices:
        eq_pitch_angles.append((pitch_angle[i] + pitch_angle[i - 1]) * 0.5)

    return eq_pitch_angles

def gca(dt, position, gyrofreq):
    b, a = signal.butter(4, np.amin(gyrofreq) / (2 * np.pi) * 0.1, fs=(1. / dt))
    zi = signal.lfilter_zi(b, a)

    x, _ = signal.lfilter(b, a, position[:, 0], zi=zi*position[0, 0])
    y, _ = signal.lfilter(b, a, position[:, 1], zi=zi*position[0, 1])
    z, _ = signal.lfilter(b, a, position[:, 2], zi=zi*position[0, 2])

    gca_trajectory = np.zeros(np.shape(position))
    gca_trajectory[:, 0] = x
    gca_trajectory[:, 1] = y
    gca_trajectory[:, 2] = z

    return gca_trajectory

diagnostics = {
    'position': {'func': Particle.r, 'requires_B': False, 'dims': 3, 'label': 'Position (m)'},
    'velocity': {'func': Particle.v, 'requires_B': False, 'dims': 3, 'label': 'Velocity (m/s)'},
    'perp_velocity': {'func': Particle.v_par, 'requires_B': True, 'dims': 1, 'label': 'Velocity (m/s)'},
    'par_velocity': {'func': Particle.v_perp, 'requires_B': True, 'dims': 1, 'label': 'Velocity (m/s)'},
    'momentum': {'func': Particle.p, 'requires_B': False, 'dims': 3, 'label': 'Momentum (kgm/s)'},
    'energy': {'func': Particle.E, 'requires_B': False, 'dims': 1, 'label': 'Energy (eV)'},
    'moment': {'func': Particle.moment, 'requires_B': True, 'dims': 1, 'label': 'Magnetic Moment (ampere*m^2)'},
    'pitch_angle': {'func': Particle.pitch_angle, 'requires_B': True, 'dims': 1, 'label': 'Pitch Angle (radians)'},
    'gyroradius': {'func': Particle.gyroradius, 'requires_B': True, 'dims': 1, 'label': 'Gyroradius (m)'},
    'gyrofreq': {'func': Particle.gyrofreq, 'requires_B': True, 'dims': 1, 'label': 'Gyrofrequency (rad/s)'},
    'b_strength': {'func': Particle.b_strength, 'requires_B': True, 'dims': 1, 'label': 'B Field (T)'}
}
