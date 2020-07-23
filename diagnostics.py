import numpy as np
from numba import njit
from scipy import constants as sp
from scipy import signal
from utils import *


@njit
def lost_ind(history):
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])

    zero_array = np.zeros((4, 3))
    index_list = np.zeros(num_particles) + steps

    for i in range(num_particles):
        for j in range(step):
            if history[i, -(j + 1), :, :] == zero_array:
                index_list[i] -= 1
            else:
                break

    return index_list

    
@njit
def position(history):
    return history[:, :, 0]


@njit
def velocity(history):
    return history[:, :, 1]


@njit
def position_mag(history):
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])

    r = history[:, :, 0, :]

    position_mags = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            position_mags[i, j] = np.sqrt(dot(r[i, j], r[i, j]))
    
    return position_mags


@njit
def b_mag(history):
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])

    b_field = history[:, :, 2, :]

    b_mags = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            b_mags[i, j] = np.sqrt(dot(b_field[i, j], b_field[i, j]))
    
    return b_mags


@njit
def velocity_par(history):
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])

    v       = history[:, :, 1]
    b_field = history[:, :, 2]

    v_vec = np.zeros((num_particles, steps, 3))
    v_mag = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            v_dot_b   = dot(v[i, j], b_field[i, j])
            b_squared = dot(b_field[i, j], b_field[i, j])
            if b_squared == 0:
                v_vec[i, j] = np.zeros(3)
                v_mag[i, j] = 0.0
                continue
            v_vec[i, j] = v_dot_b / b_squared * b_field[i, j]
            v_mag[i, j] = np.sqrt(dot(v_vec[i, j], v_vec[i, j]))

    return v_vec, v_mag


@njit
def velocity_perp(history):
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])

    velocity         = history[:, :, 1]
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
            if b_magnitude == 0:
                history_new[i, j] = 0.0
                continue
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
            aligned = np.sign(dot(v_par[i, j], history[i, j, 2, :]))
            history_new[i, j] = np.degrees(np.mod(np.arctan2(v_perp_mag, aligned * v_par_mag), np.pi))

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
def eq_pitch_angle_from_moment(history, intrinsic):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])

    mom = magnetic_moment(history, intrinsic)
    bm  = b_mag(history)
    # pa  = np.radians(pitch_angle(history))
    v   = history[:, :, 1, :]
    
    history_new = np.zeros((num_particles, steps))

    for i in range(num_particles):
        b_min = np.amin(bm[i])
        for j in range(steps):
            history_new[i, j] = np.arcsin(np.sqrt(mom[i, j] * 2 * b_min / (intrinsic[i, 0] * dot(v[i, j], v[i, j]))))
            # history_new[i, j] = np.arcsin(np.sqrt(b_min / bm[i, j]) * np.sin(pa[i, j]))
            
    return np.degrees(history_new)


@njit
def get_eq_pas(field, history, intrinsic, threshold=0.1):
    num_particles = len(history[:, 0, 0, 0])
    steps = len(history[0, :, 0, 0])
    
    eq_pa = eq_pitch_angle_from_moment(history, intrinsic)
    r = position(history)
    K = kinetic_energy(history, intrinsic)
    pas_old = np.zeros((num_particles, steps, 3)) - 1
    
    max_crossings = 0
    
    for i in range(num_particles):
        K_max = np.amax(K[i, :]) 
        ad_param = adiabaticity(field, r[i, :, :], K_max)
        contig_args = np.argwhere(ad_param <= threshold)[:, 0]
        
        if len(contig_args) == 0:
            continue

        disc_args = np.argwhere(np.diff(contig_args) != 1)[:, 0]
                
        args = np.zeros(2 * len(disc_args) + 2)
        args[0] = contig_args[0]
        for j in range(len(disc_args)):
            args[2 * j + 1] = contig_args[disc_args[j]]
            args[2 * j + 2] = contig_args[disc_args[j] + 1]
        args[-1] = contig_args[-1]
                
        vals = np.unique(args)
        dup_args = []
        count = []
        
        for v in vals:
            a = np.argwhere(args == v)[:, 0]
            if len(a) > 1:
                dup_args.append(a[0])
                count.append(len(a))
                        
        for j in range(len(dup_args)):
            total_count = 0
            for k in range(j):
                total_count += count[k]
            args = np.delete(args, np.arange(dup_args[j] - total_count, dup_args[j] + count[j] - total_count))
                    
        if int(len(args) / 2) > max_crossings:
            max_crossings = int(len(args) / 2)
        
        for j in range(int(len(args) / 2)):
            pas_old[i, j, 0] = np.mean(eq_pa[i, int(args[2 * j]):int(args[2 * j + 1])])
            pas_old[i, j, 1] = args[2 * j]
            pas_old[i, j, 2] = args[2 * j + 1]
            
    pas_new = np.zeros((num_particles, max_crossings, 3)) - 1
    pas_new[:, :, :] = pas_old[:, 0:max_crossings, :]
                
    return pas_new


def get_pas_at_bounce_phase(history, phase):
    pas = pitch_angle(history)
    
    bounce_pas = []
    
    nth_crossing = int(phase // np.pi)
    additional_phase = phase % np.pi

    for i in range(len(pas[:, 0])):
        zero_crossings = np.where(np.diff(np.sign(pas[i] - 90)))[0]
        max_crossings = len(zero_crossings) - 1
        
        if max_crossings >= nth_crossing + 1:
            diff = zero_crossings[nth_crossing + 1] - zero_crossings[nth_crossing]
            eq_point = np.abs(pas[i, zero_crossings[nth_crossing]:zero_crossings[nth_crossing] + diff] - 90).argmax()
            first_half = eq_point
            second_half = diff - eq_point

            if additional_phase <= np.pi / 2:
                ind = int(first_half / (np.pi / 2) * additional_phase) + zero_crossings[nth_crossing]
                bounce_pas.append(pas[i, ind])
            else:
                ind = int(first_half + second_half / (np.pi / 2) * (additional_phase - np.pi / 2)) + zero_crossings[nth_crossing]
                bounce_pas.append(pas[i, ind])
    
    return bounce_pas


def get_pas_at_bounce_phase_all_t(history, phase):
    all_pas = get_pas_at_bounce_phase(history, phase)
    
    while True:
        phase = phase + 2 * np.pi
        new_pas = get_pas_at_bounce_phase(history, phase)

        if len(new_pas) == 0:
            break
        else:
            for pa in new_pas:
                all_pas.append(pa)
                
    return all_pas


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
