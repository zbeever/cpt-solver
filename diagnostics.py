import numpy as np
from math import sqrt, acos
from scipy import constants as sp
from numba import njit, prange


@njit(parallel=True)
def mag(quantity):
    '''
    Calculates the magnitude of a quantity of a history. Useful to find the distance from the origin and the strength of the magnetic field.

    Parameters
    ----------
    quantity (NxMx3 numpy array): A history of the quantity. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    mag_v (NxM numpy array): The magnitude of the quantity at each timestep for each particle.
    '''

    num_particles = np.shape(quantity)[0]
    steps = np.shape(quantity)[1]

    mag_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            mag_v[i, j] = sqrt(quantity[i, j, 0]**2 + quantity[i, j, 1]**2 + quantity[i, j, 2]**2)

    return mag_v


@njit(parallel=True)
def v_par(velocity, b_field):
    '''
    Calculates the magnitude of the velocity parallel to the magnetic field (in m/s) of a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    v_par_v (NxM numpy array): The velocity parallel to the background magnetic field at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    v_par_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            b_dot_b = b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2
            if b_dot_b == 0:
                continue

            v_dot_b = velocity[i, j, 0] * b_field[i, j, 0] + velocity[i, j, 1] * b_field[i, j, 1] + velocity[i, j, 2] * b_field[i, j, 2]
            v_par_vec = v_dot_b / b_dot_b * b_field[i, j]
            v_par_v[i, j] = sqrt(v_par_vec[0]**2 + v_par_vec[1]**2 + v_par_vec[2]**2)

    return v_par_v


@njit(parallel=True)
def v_perp(velocity, b_field):
    '''
    Calculates the magnitude of the velocity perpendicular to the magnetic field (in m/s) of a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    v_perp_v (NxM numpy array): The velocity perpendicular to the background magnetic field at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    v_perp_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            b_dot_b = b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2
            if b_dot_b == 0:
                continue

            v_dot_b = velocity[i, j, 0] * b_field[i, j, 0] + velocity[i, j, 1] * b_field[i, j, 1] + velocity[i, j, 2] * b_field[i, j, 2]
            v_perp_vec = velocity[i, j] - v_dot_b / b_dot_b * b_field[i, j]
            v_perp_v[i, j] = sqrt(v_perp_vec[0]**2 + v_perp_vec[1]**2 + v_perp_vec[2]**2)

    return v_perp_v


@njit(parallel=True)
def ke(velocity, mass):
    '''
    Calculates the kinetic energy (in eV) along of a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.

    Returns
    -------
    ke_v (M numpy array): The kinetic energy (in eV) at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    ke_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            v_dot_v = velocity[i, j, 0]**2 + velocity[i, j, 1]**2 + velocity[i, j, 2]**2
            gamma = 1.0 / sqrt(1.0 - v_dot_v / sp.c**2)
            ke_v[i, j] = mass[i] * sp.c**2 * (gamma - 1.0) * 6.24150913e18

    return ke_v


@njit(parallel=True)
def moment(velocity, b_field, mass):
    '''
    Calculates the magnetic moment (in MeV/G) along a history.

    Parameters
    ----------
    v_perp (NxM numpy array): A history of particle velocities perpendicular to the background magnetic field. The first index denotes the particle and the second the timestep.
    b_magnitude (NxM numpy array): A history of background magnetic field strengths. The first index denotes the particle and the second the timestep.

    Returns
    -------
    moment_v (NxM numpy array): The magnetic moment (in MeV/G) at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    moment_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            b_dot_b = b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2
            if b_dot_b == 0:
                continue

            v_dot_b = velocity[i, j, 0] * b_field[i, j, 0] + velocity[i, j, 1] * b_field[i, j, 1] + velocity[i, j, 2] * b_field[i, j, 2]
            v_perp_vec = velocity[i, j] - v_dot_b / b_dot_b * b_field[i, j]
            v_perp = sqrt(v_perp_vec[0]**2 + v_perp_vec[1]**2 + v_perp_vec[2]**2)

            moment_v[i, j] = 0.5 * mass[i] * v_perp**2 / sqrt(b_dot_b) * 6.242e8

    return moment_v


@njit(parallel=True)
def pitch_ang(velocity, b_field):
    '''
    Calculates the pitch angle (in radians) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    pitch_ang_v (NxM numpy array): The pitch angle (in radians) at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    pitch_ang_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            b_mag = sqrt(b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2)
            if b_mag == 0:
                continue

            v_dot_b = velocity[i, j, 0] * b_field[i, j, 0] + velocity[i, j, 1] * b_field[i, j, 1] + velocity[i, j, 2] * b_field[i, j, 2]
            v_mag = sqrt(velocity[i, j, 0]**2 + velocity[i, j, 1]**2 + velocity[i, j, 2]**2)

            pitch_ang_v[i, j] = acos(v_dot_b / (v_mag * b_mag))

    return pitch_ang_v


@njit(parallel=True)
def gyrorad(velocity, b_field, mass, charge):
    '''
    Calculates the gyroradius (in m) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.
    charge (M numpy array): A list of particle charges.

    Returns
    -------
    gyrorad_v (NxM numpy array): The gyroradius (in m) at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    gyrorad_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            b_dot_b = b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2
            if b_dot_b == 0:
                continue

            v_dot_b = velocity[i, j, 0] * b_field[i, j, 0] + velocity[i, j, 1] * b_field[i, j, 1] + velocity[i, j, 2] * b_field[i, j, 2]
            v_dot_v = velocity[i, j, 0]**2 + velocity[i, j, 1]**2 + velocity[i, j, 2]**2

            gamma = 1.0 / sqrt(1.0 - v_dot_v / sp.c**2)
            b_mag = sqrt(b_dot_b)

            v_perp_vec = velocity[i, j] - v_dot_b / b_dot_b * b_field[i, j]
            v_perp = sqrt(v_perp_vec[0]**2 + v_perp_vec[1]**2 + v_perp_vec[2]**2)

            gyrorad_v[i, j] = gamma * mass[i] * v_perp / (abs(charge[i]) * b_mag)

    return gyrorad_v


@njit(parallel=True)
def gyrofreq(velocity, b_field, mass, charge):
    '''
    Calculates the gyrofrequency (in 1/s) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.
    charge (M numpy array): A list of particle charges.

    Returns
    -------
    gyrofreq_v (NxM numpy array): The gyrofrequency (in 1/s) at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps = np.shape(velocity)[1]

    gyrofreq_v = np.zeros((num_particles, steps))

    for i in prange(num_particles):
        for j in prange(steps):
            b_mag = sqrt(b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2)
            if b_mag == 0:
                continue

            v_dot_v = velocity[i, j, 0]**2 + velocity[i, j, 1]**2 + velocity[i, j, 2]**2
            gamma = 1.0 / sqrt(1.0 - v_dot_v / sp.c**2)

            gyrofreq_v[i, j] = abs(charge[i]) * b_mag / (2 * np.pi * gamma * mass[i])
    
    return gyrofreq_v 


@njit(parallel=True)
def gca(b_field, position, velocity, mass, charge):
    '''
    Find (roughly) the guiding center trajectory along a history. This function adapted from Kaan Ozturk's RAPT code: https://github.com/mkozturk/rapt/

    Parameters
    ----------
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.
    position (NxMx3 numpy array): A history of particle locations. The first index denotes the particle, the second the timestep, and the third the dimension.
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.
    charge (N numpy array): A list of particle charges.

    Returns
    -------
    gca_v (NxMx3 numpy array): The guiding center trajectory for each particle.
    '''

    num_particles = np.shape(position)[0]
    steps = np.shape(position)[1]

    gca_v = np.zeros((num_particles, steps, 3))

    for i in prange(num_particles):
        for j in prange(steps):
            b_dot_b = b_field[i, j, 0]**2 + b_field[i, j, 1]**2 + b_field[i, j, 2]**2
            if b_dot_b == 0:
                continue
            
            b_cross_v = np.zeros(3)
            b_cross_v[0] = b_field[i, j, 1] * velocity[i, j, 2] - b_field[i, j, 2] * velocity[i, j, 1]
            b_cross_v[1] = b_field[i, j, 2] * velocity[i, j, 0] - b_field[i, j, 0] * velocity[i, j, 2]
            b_cross_v[2] = b_field[i, j, 0] * velocity[i, j, 1] - b_field[i, j, 1] * velocity[i, j, 0]

            v_dot_v = velocity[i, j, 0]**2 + velocity[i, j, 1]**2 + velocity[i, j, 2]**2
            gamma = 1.0 / sqrt(1.0 - v_dot_v / sp.c**2)

            gca_v[i, j] = position[i, j] - gamma * mass[i] / (charge[i] * b_dot_b) * b_cross_v
             
    return gca_v


@njit(parallel=True)
def diffusion(quantity, time, delta_t, bins=100):
    '''
    Calculate the diffusion coefficient of a quantity along a history. This value is indexed by bin number,
    where bins are uniformly distributed between the minimum and maximum values of the given quantity.

    Parameters
    ----------
    quantity (NxM numpy array): A history of the quantity to use. The first index denotes the particle and the second the timestep.
    time (M numpy array): An array of timesteps associated with the history.
    delta_t (float): The timestep over which diffusion will be calculated.
    bins (int): The number of bins to use. Defaults to 100.

    Returns
    -------
    bins_v (BINS numpy array): The bin labels.
    diffusion_v (BINSxM numpy array): The diffusion coefficient at each timestep.
    '''

    num_particles = np.shape(quantity)[1]
    steps = np.shape(quantity)[1]
    
    dt = np.abs(time[1] - time[0])
    delta_t_ind = int(max(delta_t // dt, 1))
    inv_diff_time = 0.5 / (delta_t_ind * dt)
    
    max_val = np.amax(quantity)
    min_val = np.amin(quantity)
    bin_width = (max_val - min_val) / (bins - 1)
    
    unweighted_diff_coef = np.zeros((bins, steps))
    weights = np.zeros((bins, steps))
    
    for i in prange(num_particles):
        for j in prange(steps - delta_t_ind - 1):
            ind_diff_coef = (quantity[i, j + delta_t_ind] - quantity[i, j])**2 * inv_diff_time
            bin_ind = int((quantity[i, j] + min_val) // bin_width)
            unweighted_diff_coef[bin_ind, j] += ind_diff_coef
            weights[bin_ind, j] += 1
            
    bins_v = np.linspace(0, max_val, bins)
    diffusion_v = np.zeros((bins, steps))
            
    for i in prange(bins):
        for j in prange(steps - delta_t_ind - 1):
            if weights[i, j] != 0:
                diffusion_v[i, j] = unweighted_diff_coef[i, j] / weights[i, j]
    
    return bins_v, diffusion_v


@njit(parallel=True)
def transport(quantity, time, delta_t, bins=100):
    '''
    Calculate the transport coefficient of a quantity (its average change in a given time) along a history.
    This value is indexed by bin number, where bins are uniformly distributed between the minimum and maximum
    values of the given quantity.

    Parameters
    ----------
    quantity (NxM numpy array): A history of the quantity to use. The first index denotes the particle and the second the timestep.
    time (M numpy array): An array of timesteps associated with the history.
    delta_t (float): The timestep over which transport will be calculated.
    bins (int): The number of bins to use. Defaults to 100.

    Returns
    -------
    bins_v (BINS numpy array): The bin labels.
    diffusion_v (BINSxM numpy array): The transport coefficient at each timestep.
    '''

    num_particles = np.shape(quantity)[1]
    steps = np.shape(quantity)[1]
    
    dt = np.abs(time[1] - time[0])
    delta_t_ind = int(max(delta_t // dt, 1))
    inv_diff_time = 0.5 / (delta_t_ind * dt)
    
    max_val = np.amax(quantity)
    min_val = np.amin(quantity)
    bin_width = (max_val - min_val) / (bins - 1)
    
    unweighted_trans_coef = np.zeros((bins, steps))
    weights = np.zeros((bins, steps))
    
    for i in prange(num_particles):
        for j in prange(steps - delta_t_ind - 1):
            ind_trans_coef = quantity[i, j + delta_t_ind] - quantity[i, j]
            bin_ind = int((quantity[i, j] + min_val) // bin_width)
            unweighted_trans_coef[bin_ind, j] += ind_trans_coef
            weights[bin_ind, j] += 1
            
    bins_v = np.linspace(0, max_val, bins)
    transport_v = np.zeros((bins, steps))
            
    for i in prange(bins):
        for j in prange(steps - delta_t_ind - 1):
            if weights[i, j] != 0:
                transport_v[i, j] = unweighted_trans_coef[i, j] / weights[i, j]
    
    return bins_v, transport_v
