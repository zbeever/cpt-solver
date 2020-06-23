import numpy as np
from numba import njit
from scipy import constants as sp
from scipy import signal
from utils import *

@njit
def position(history):
    return history[:, :, 0]


@njit
def position_mag(history):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    r = position(history)
    history_new = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            history_new[i, j] = np.sqrt(dot(r[i, j], r[i, j]))
    
    return history_new


@njit
def velocity(history):
    return history[:, :, 1]


@njit
def b_mag(history):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    b_field = history[:, :, 2]
    history_new = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            history_new[i, j] = np.sqrt(dot(b_field[i, j], b_field[i, j]))
    
    return history_new


@njit
def velocity_par(history):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    v = history[:, :, 1]
    b_field = history[:, :, 2]

    history_new = np.zeros((num_particles, steps, 3))

    for i in range(num_particles):
        for j in range(steps):
            v_dot_b = dot(v[i, j], b_field[i, j])
            b_squared = dot(b_field[i, j], b_field[i, j])
            history_new[i, j] = v_dot_b / b_squared * b_field[i, j]

    return history_new


@njit
def velocity_perp(history):
    velocity = history[:, :, 1]
    return velocity - velocity_par(history)


@njit
def kinetic_energy(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    v = history[:, :, 1]
    history_new = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            gamma_v = gamma(v[i, j])
            history_new[i, j] = J_to_eV(intrinsic[i, 0] * sp.c ** 2 * (gamma_v - 1.0))

    return history_new


@njit
def magnetic_moment(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    v_perp = velocity_perp(history)
    b_field = history[:, :, 2]
    history_new = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            v_perp_squared = dot(v_perp[i, j], v_perp[i, j])
            b_magnitude = np.sqrt(dot(b_field[i, j], b_field[i, j]))
            history_new[i, j] = 0.5 * intrinsic[i, 0] * v_perp_squared / b_magnitude

    return history_new


@njit
def pitch_angle(history):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    v_par = velocity_par(history)
    v_perp = history[:, :, 1] - v_par
    history_new = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            v_par_mag = np.sqrt(dot(v_par[i, j],  v_par[i, j]))
            v_perp_mag = np.sqrt(dot(v_perp[i, j], v_perp[i, j]))
            history_new[i, j] = np.degrees(np.mod(np.arctan2(v_perp_mag, v_par_mag), np.pi))

    return history_new


@njit
def gyrorad(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    v = history[:, :, 1]
    v_perp = velocity_perp(history)
    b_field = history[:, :, 2]
    history_new = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            gamma_v = gamma(v[i, j])
            v_perp_mag = np.sqrt(dot(v_perp[i, j], v_perp[i, j]))
            b_magnitude = np.sqrt(dot(b_field[i, j], b_field[i, j]))
            history_new[i, j] = gamma_v * intrinsic[i, 0] * v_perp_mag / (abs(intrinsic[i, 1]) * b_magnitude)
    
    return history_new 


@njit
def gyrofreq(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    v = history[:, :, 1]
    b_field = history[:, :, 2]
    history_new = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            gamma_v = gamma(v[i, j])
            b_magnitude = np.sqrt(dot(b_field[i, j], b_field[i, j]))
            history_new[i, j] = abs(intrinsic[i, 1]) * b_magnitude / (gamma_v * intrinsic[i, 0])

    return history_new


@njit
def eq_pitch_angle(history):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    z = history[:, :, 0, 2]
    z_sign = np.sign(z)
    pitch_angles = pitch_angle(history)

    history_new = np.zeros((num_particles, steps))

    for i in range(num_particles):
        equatorial_crossings = ((np.roll(z_sign[i, :], 1) - z_sign[i, :]) != 0)
        equatorial_crossings[0] = False
        k = 0
        for j in range(len(equatorial_crossings)):
            if equatorial_crossings[j] == True:
                z0 = z[i, j - 1]
                z1 = z[i, j]
                p0 = pitch_angles[i, j - 1]
                p1 = pitch_angles[i, j]
                history_new[i, k] = ((p0 - p1) / (z1 - z0)) * z0 + p0
                k += 1
    
    return history_new


@njit
def gca_nonrel(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    r = history[:, :, 0]
    v = history[:, :, 1]
    m = intrinsic[:, 0]
    q = intrinsic[:, 1]
    B = history[:, :, 2, :]
    E = history[:, :, 3, :]

    history_new =  np.zeros((num_particles, steps, 3))
    
    for i in range(num_particles):
        for j in range(steps):
            rho = m[i] / (q[i] * dot(B[i, j], B[i, j])) * np.cross(B[i, j], v[i, j] - np.cross(E[i, j], B[i, j]) / dot(B[i, j], B[i, j]))
            history_new[i, j] = r[i, j] - rho
            
    return history_new


def gca_filter(history, intrinsic, dt):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    position = history[:, :, 0]
    gyrofrequency_list = gyrofreq(history, intrinsic)

    history_new =  np.zeros((num_particles, steps, 3))
    
    for i in range(num_particles):
        b, a = signal.butter(4, np.amin(gyrofrequency_list[i]) / (2 * np.pi) * 0.1, fs=(1. / dt))
        zi = signal.lfilter_zi(b, a)

        x, _ = signal.lfilter(b, a, position[i, :, 0], zi=zi*position[i, 0, 0])
        y, _ = signal.lfilter(b, a, position[i, :, 1], zi=zi*position[i, 0, 1])
        z, _ = signal.lfilter(b, a, position[i, :, 2], zi=zi*position[i, 0, 2])

        history_new[i, :, 0] = x
        history_new[i, :, 1] = y
        history_new[i, :, 2] = z

    return history_new
