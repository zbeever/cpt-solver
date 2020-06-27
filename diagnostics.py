import numpy as np
from constants import *
from numba import njit
from scipy import constants as sp
from scipy import signal
from utils import *


@njit
def gamma(v):
    return 1.0 / np.sqrt(1 - dot(v, v) / sp.c**2)


@njit
def dot(v, w):
    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]


@njit
def J_to_eV(E):
    return 1.0 / sp.e * E


@njit
def eV_to_J(E):
    return sp.e * E


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

    v_vec = np.zeros((num_particles, steps, 3))
    v_mag = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            v_dot_b = dot(v[i, j], b_field[i, j])
            b_squared = dot(b_field[i, j], b_field[i, j])
            v_vec[i, j] = v_dot_b / b_squared * b_field[i, j]
            v_mag[i, j] = np.sqrt(dot(v_vec[i, j], v_vec[i, j]))

    return v_vec, v_mag


@njit
def velocity_perp(history):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    velocity = history[:, :, 1]
    v_par, v_par_mag = velocity_par(history)

    v_vec = velocity - v_par
    v_mag = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            v_mag[i, j] = np.sqrt(dot(v_vec[i, j], v_vec[i, j]))

    return v_vec, v_mag


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

    v_perp, v_perp_mag = velocity_perp(history)
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

    v_par, v_par_mag = velocity_par(history)
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
    v_perp, v_perp_mag = velocity_perp(history)
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
def eq_pitch_angle_from_moment(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    mom = magnetic_moment(history, intrinsic)
    bm  = b_mag(history)
    v   = history[:, :, 1, :]
    
    history_new = np.zeros((num_particles, steps))

    for i in range(num_particles):
        b_min = np.amin(bm[i])
        for j in range(steps):
            history_new[i, j] = np.arcsin(np.sqrt(mom[i, j] * 2 * b_min / (intrinsic[i, 0] * dot(v[i, j], v[i, j]))))
            
    return np.degrees(history_new)


def get_eq_pas(history, intrinsic, dt, threshold=0.2, min_time=5e-3, padding=1e-3):
    eq_pa_hist = eq_pitch_angle_from_moment(history, intrinsic)
    
    all_eq_pas = np.zeros((len(history[:, 0, 0, 0]), len(history[0, :, 0, 0]), 3)) - 1
    
    for j in range(len(eq_pa_hist[:, 0])):
        centered = np.diff(eq_pa_hist[j, :], prepend=eq_pa_hist[j, 0])
        within_thresh = np.argwhere(np.abs(centered) <= threshold)[:, 0]

        contiguous = np.diff(within_thresh, prepend=within_thresh[0])

        endpoints = []

        pad = int(min_time / dt)
        stretch = int(padding / dt)

        endpoints = []
        endpoints_tentative = np.where(np.concatenate(([contiguous[0]], contiguous[:-1] != contiguous[1:], [1])))[0] - 1
        for i in range(int(len(endpoints_tentative) / 2)):
            if ((endpoints_tentative[2 * i + 1] - pad) - (endpoints_tentative[2 * i] + pad)) > stretch:
                endpoints.append(endpoints_tentative[2 * i] + pad)
                endpoints.append(endpoints_tentative[2 * i + 1] - pad)

        for i in range(int(len(endpoints) / 2)):
            all_eq_pas[j, i, 0] = np.mean(eq_pa_hist[j, :][within_thresh][endpoints[2 * i]:endpoints[2 * i + 1]])
            all_eq_pas[j, i, 1] = within_thresh[endpoints[2 * i]]
            all_eq_pas[j, i, 2] = within_thresh[endpoints[2 * i + 1]]
    
    k = 1
    for i in range(len(all_eq_pas[:, 0, 0])):
        j = 1
        while all_eq_pas[i, j - 1, 0] != -1:
            j += 1
        if j > k:
            k = j
        
    new_eq_pas = np.zeros((len(history[:, 0, 0, 0]), k, 3)) - 1
    new_eq_pas = all_eq_pas[:, 0:k, :]
        
    return new_eq_pas


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
