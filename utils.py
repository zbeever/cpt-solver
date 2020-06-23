import numpy as np
from numba import njit
import scipy.constants as sp

axis_num = {
        'x': 0,
        'y': 1,
        'z': 2
        }

@njit
def J_to_eV(E):
    return 1.0 / sp.e * E


@njit
def eV_to_J(E):
    return sp.e * E


@njit
def dot(v, w):
    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]


@njit
def gamma(v):
    return 1.0 / np.sqrt(1 - dot(v, v) / sp.c**2)


@njit
def local_onb(r, b_field, t = 0.):
    B = b_field(r, t)

    local_z = B
    if np.dot(local_z, local_z) == 0:
        local_z = np.array([0., 0., 1.])
    else:
        local_z = local_z / np.linalg.norm(local_z)

    local_x = -r
    local_x = local_x - np.dot(local_x, local_z) * local_z
    if np.dot(local_x, local_x) == 0:
        local_x = np.array([-1., 0., 0.])
    else:
        local_x = local_x / np.linalg.norm(local_x)

    local_y = np.cross(local_z, local_x)
    return local_x, local_y, local_z


@njit
def velocity_vec(r, K, m, b_field, pitch_angle, phase_angle, t = 0.):
    local_x, local_y, local_z = local_onb(r, b_field, t)

    v_dir = np.sin(pitch_angle) * np.cos(phase_angle) * local_x + np.sin(pitch_angle) * np.sin(phase_angle) * local_y + np.cos(pitch_angle) * local_z

    gamma_v = eV_to_J(K) / (m * sp.c ** 2.0) + 1.0
    v_mag = sp.c * np.sqrt(1. - gamma_v ** (-2.0))

    if np.dot(v_dir, v_dir) == 0.0:
        return np.array([0.0, 0.0, 0.0])
    else:
        return v_dir / np.linalg.norm(v_dir) * v_mag
