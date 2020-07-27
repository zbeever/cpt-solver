import numpy as np
from scipy import constants as sp
from scipy import signal
from numba import njit

from utils import dot, J_to_eV, eV_to_J, gamma, flc, gyrovector


@njit
def position_mag(position):
    '''
    Calculates the distance from the origin (in m) along a history.

    Parameters
    ----------
    position (NxMx3 numpy array): A history of particle locations. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    r_mag_v (NxM numpy array): The distance from the origin at each timestep for each particle.
    '''

    num_particles = np.shape(position)[0]
    steps         = np.shape(position)[1]

    r_mag_v = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            r_mag_v[i, j] = np.linalg.norm(position[i, j])
    
    return r_mag_v


@njit
def b_mag(b_field):
    '''
    Calculates the magnetic field strength (in T) along a history.

    Parameters
    ----------
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    b_mag_v (NxM numpy array): The background magnetic field strength at each timestep for each particle.
    '''

    num_particles = np.shape(b_field)[0]
    steps         = np.shape(b_field)[1]

    b_mag_v = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            b_mag_v[i, j] = np.linalg.norm(b_field[i, j])
    
    return b_mag_v


@njit
def velocity_par(velocity, b_field):
    '''
    Calculates the magnitude of the velocity parallel to the magnetic field (in m/s) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    v_par_v (NxM numpy array): The velocity parallel to the background magnetic field at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0] 
    steps         = np.shape(velocity)[1]

    v_par_v = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            v_dot_b   = dot(velocity[i, j], b_field[i, j])
            b_squared = dot(b_field[i, j], b_field[i, j])

            if b_squared == 0:
                v_par_v[i, j] = 0.0
                continue

            v_par_vec     = v_dot_b / b_squared * b_field[i, j]
            v_par_v[i, j] = np.linalg.norm(v_par_vec)

    return v_par_v


@njit
def velocity_perp(velocity, b_field):
    '''
    Calculates the magnitude of the velocity perpendicular to the magnetic field (in m/s) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.

    Returns
    -------
    v_perp_v (NxM numpy array): The velocity perpendicular to the background magnetic field at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps         = np.shape(velocity)[1]

    v_perp_v = np.zeros((num_particles, steps))

    for i in range(num_particles):
        for j in range(steps):
            v_dot_b   = dot(velocity[i, j], b_field[i, j])
            b_squared = dot(b_field[i, j], b_field[i, j])

            if b_squared == 0:
                v_perp_v[i, j] = np.linalg.norm(velocity[i, j])
                continue

            v_perp_vec     = velocity[i, j] - v_dot_b / b_squared * b_field[i, j]
            v_perp_v[i, j] = np.linalg.norm(v_perp_vec)

    return v_perp_v


@njit
def kinetic_energy(velocity, mass):
    '''
    Calculates the kinetic energy (in eV) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.

    Returns
    -------
    ke_v (NxM numpy array): The kinetic energy (in eV) at each timestep for each particle.
    '''

    num_particles = np.shape(velocity)[0]
    steps         = np.shape(velocity)[1]

    ke_v = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            gamma_v = gamma(velocity[i, j])
            ke_v[i, j] = J_to_eV(mass[i] * sp.c**2 * (gamma_v - 1.0))

    return ke_v


@njit
def magnetic_moment(v_perp, b_magnitude, mass):
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

    num_particles = np.shape(v_perp)[0]
    steps         = np.shape(v_perp)[1]

    moment_v = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            if b_magnitude[i, j] == 0:
                moment_v[i, j] = 0.0
                continue
            moment_v[i, j] = 0.5 * mass[i] * v_perp[i, j]**2 / b_magnitude[i, j] * (6.242e8)

    return moment_v


@njit
def pitch_angle(velocity, b_field):
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
    steps         = np.shape(velocity)[1]

    pitch_ang_v = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            v_mag     = np.linalg.norm(velocity[i, j])
            b_mag     = np.linalg.norm(b_field[i, j])
            v_dot_b   = dot(velocity[i, j], b_field[i, j])
            cos_theta = v_dot_b / (v_mag * b_mag)

            pitch_ang_v[i, j] = np.arccos(cos_theta)

    return pitch_ang_v


@njit
def gyrorad(ke, v_perp_magnitude, b_magnitude, mass, charge):
    '''
    Calculates the gyroradius (in m) along a history.

    Parameters
    ----------
    ke (NxM numpy array): A history of particle kinetic energies. The first index denotes the particle and the second the timestep.
    v_perp_magnitude (NxM numpy array): A history of velocities perpendicular to the background magnetic field. The first index denotes the particle and the second the timestep.
    b_magnitude (NxM numpy array): A history of background magnetic field strengths. The first index denotes the particle and the second the timestep.
    mass (N numpy array): A list of particle masses.
    charge (N numpy array): A list of particle charges.

    Returns
    -------
    gyrorad_v (NxM numpy array): The gyroradius (in m) at each timestep for each particle.
    '''

    num_particles = np.shape(ke)[0]
    steps         = np.shape(ke)[1]

    gyrorad_v = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            gamma_v = ke[i, j] / J_to_eV(mass[i] * sp.c**2) + 1
            gyrorad_v[i, j] = gamma_v * mass[i] * v_perp_magnitude[i, j] / (abs(charge[i]) * b_magnitude[i, j])
    
    return gyrorad_v 


@njit
def gyrofreq(ke, b_magnitude, mass, charge):
    '''
    Calculates the gyrofrequency (in 1/s) along a history.

    Parameters
    ----------
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    b_field (NxMx3 numpy array): A history of background magnetic field vectors. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.
    charge (N numpy array): A list of particle charges.

    Returns
    -------
    gyrofreq_v (NxM numpy array): The gyrofrequency (in 1/s) at each timestep for each particle.
    '''

    num_particles = np.shape(ke)[0]
    steps         = np.shape(ke)[1]

    gyrofreq_v = np.zeros((num_particles, steps)) 

    for i in range(num_particles):
        for j in range(steps):
            gamma_v = ke[i, j] / J_to_eV(mass[i] * sp.c**2) + 1
            gyrofreq_v[i, j] = abs(charge[i]) * b_magnitude[i, j] / (2 * np.pi * gamma_v * mass[i])

    return gyrofreq_v


def eq_pitch_angle(b_field, pitch_ang, b_magnitude, gyrorad, position, unwrapped=False):
    '''
    Calculates the equatorial pitch angle (in radians) along a history. Does so by taking advantage of the fact that the first adiabatic invariant
    is most accurate at the mirror points along a particle's trajecotry. A single equatorial pitch angle is assigned to the interval between adjacent
    equatorial crossing points.

    Parameters
    ----------
    b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py). This is required for checking adiabaticity.
    pitch_ang (NxM numpy array): A history of particle pitch angles. The first index denotes the particle and the second the timestep.
    b_magnitude (NxM numpy array): A history of background magnetic field strengths. The first index denotes the particle and the second the timestep.
    gyrorad (NxM numpy array): A history of gyroradii. The first index denotes the particle and the second the timestep.
    position (NxMx3 numpy array): A history of particle locations. The first index denotes the particle, the second the timestep, and the third the dimension.
    unwrapped (bool): Whether the equatorial pitch angle should be displayed from 0 to pi / 2 or unwrapped and displayed from 0 to pi.

    Returns
    -------
    eq_pa_v (NxM numpy array): The equatorial pitch angle (in radians) at each timestep for each particle.
    '''

    num_particles = np.shape(pitch_ang)[0]
    steps         = np.shape(pitch_ang)[1]

    eps = 2e-1
    
    eq_pa_v = np.zeros(np.shape(pitch_ang))

    for i in range(num_particles):
        # Find mirror points (where the particle has a pitch angle of 90 degrees).
        mp_ind = np.argwhere(np.diff(np.sign(pitch_ang[i, :] - np.pi / 2)) != 0)[:, 0]

        # Find crossing points (where the particle encounters the minimum magnetic field strength).
        x_ind = np.zeros(len(mp_ind) + 1, dtype=int)
        
        # Search before the first mirror point
        if mp_ind[0] != 0:
            x_ind[0] = b_magnitude[i, 0:mp_ind[0]].argmin()
        # Search between the mirror points
        for k in range(len(mp_ind) - 1):
            x_ind[k + 1] = b_magnitude[i, mp_ind[k]:mp_ind[k + 1]].argmin() + mp_ind[k]
        # Search after the last mirror point
        if mp_ind[-1] != steps - 1:
            x_ind[-1] = b_magnitude[i, mp_ind[-1]:].argmin() + mp_ind[-1]

        # Remove crossing points that do not respresent actual crossing. We can identify these
        # by comparing the distance from each mirror point to its two adjacent crossing points.
        # For the crossing points to represent legitimate crossings, they must be symmetrical with
        # respect to the mirror point.
        asymmetry_beg = 10 * eps
        asymmetry_end = 10 * eps
        if x_ind[1] != mp_ind[0]:
            asymmetry_beg = np.abs((x_ind[1] - mp_ind[0]) - (mp_ind[0] - x_ind[0])) / np.abs(2 * (x_ind[1] - mp_ind[0]))
        if mp_ind[-1] != x_ind[-2]:
            asymmetry_end = np.abs((x_ind[-1] - mp_ind[-1]) - (mp_ind[-1] - x_ind[-2])) / np.abs(2 * (mp_ind[-1] - x_ind[-2]))
        x_ind_fixed = []
        if asymmetry_beg > eps and asymmetry_end > eps:
            x_ind_fixed = x_ind[1:-1]
        elif asymmetry_beg > eps and asymmetry_end <= eps:
            x_ind_fixed = x_ind[1:]
        elif asymmetry_beg <= eps and asymmetry_end > eps:
            x_ind_fixed = x_ind[:-1]
        else:
            x_ind_fixed = x_ind
        x_ind_fixed = np.array(x_ind_fixed)

        # Add mirror points at the beginning and end of the trajectory that were not detected.
        if x_ind_fixed[0] > 0:
            closest_pa_ind = np.abs(pitch_ang[i, 0:x_ind_fixed[0]] - np.pi / 2).argmin()
            if np.abs(pitch_ang[i, closest_pa_ind] - np.pi / 2) <= eps and closest_pa_ind != mp_ind[0]:
                if mp_ind[0] > x_ind_fixed[0]:
                    mp_ind = np.append(closest_pa_ind, mp_ind)
        if x_ind_fixed[-1] < steps - 1:
            closest_pa_ind = np.abs(pitch_ang[i, x_ind_fixed[-1]:] - np.pi / 2).argmin()  + x_ind_fixed[-1]
            if np.abs(pitch_ang[i, closest_pa_ind] - np.pi / 2) <= eps and closest_pa_ind != mp_ind[-1]:
                if mp_ind[-1] < x_ind_fixed[-1]:
                    mp_ind = np.append(mp_ind, closest_pa_ind)

        # Same as above, but in this case we look for a high adiabaticity in place of a true mirror point.
        if x_ind_fixed[0] < mp_ind[0]:
            R_c = flc(b_field, position[i, 0])
            if gyrorad[i, 0] / R_c < 1e-3:
                mp_ind = np.append(0, mp_ind)
        if x_ind_fixed[-1] > mp_ind[-1]:
            R_c = flc(b_field, position[i, -1])
            if gyrorad[i, -1] / R_c < 1e-3:
                mp_ind = np.append(mp_ind, steps - 1)

        # We assign an equatorial pitch angle to each interval between two mirror points.
        # To accomodate situations where there is no mirror point before or after a crossing
        # point, we pad the beginning and ending of the (cosmetic) crossing array with extremal values.
        display_xs = np.copy(x_ind_fixed)
        if x_ind_fixed[0] <= mp_ind[0]:
            display_xs[0] = 0
        else:
            display_xs = np.append(0, display_xs)
        if x_ind_fixed[-1] > mp_ind[-1]:
            display_xs[-1] = steps - 1
        else:
            display_xs = np.append(display_xs, steps - 1)

        # For each mirror point, calculate the equatorial pitch angle and assign it to the
        # time interval between its adjacent crossing points.
        for k in range(len(mp_ind)):
            b_mirror = b_magnitude[i, mp_ind[k]]
            b_min = b_magnitude[i, x_ind_fixed[min(k, len(x_ind_fixed) - 1)]]
            eq_pa_sin = np.sqrt(b_min / b_mirror) * np.sin(pitch_ang[i, mp_ind[k]])
            # Rarely, in scenarios with equatorial pitch angles near 90 degrees, this function will find a mirror point
            # with a weaker magnetic field strength than its crossing point. In this case, we set the equatorial pitch angle
            # to be 90 degrees.
            if eq_pa_sin > 1:
                eq_pa_v[i, display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1] = np.pi / 2
            elif unwrapped is True:
                eq_pa_v[i, display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1] = np.mod(np.sign(np.pi / 2 - pitch_ang[i, display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1]) * np.arcsin(eq_pa_sin), np.pi)
            else:
                eq_pa_v[i, display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1] = np.arcsin(eq_pa_sin)
            
    return eq_pa_v


@njit
def gca(b_field, position, velocity, mass, charge, max_iterations=20, tolerance=1e-3):
    '''
    Iteratively find the guiding center trajectory along a history. This function adapted from Kaan Ozturk's RAPT code: https://github.com/mkozturk/rapt/

    Parameters
    ----------
    b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py).
    position (NxMx3 numpy array): A history of particle locations. The first index denotes the particle, the second the timestep, and the third the dimension.
    velocity (NxMx3 numpy array): A history of particle velocities. The first index denotes the particle, the second the timestep, and the third the dimension.
    mass (N numpy array): A list of particle masses.
    charge (N numpy array): A list of particle charges.
    max_iterations (int): The maximum number of iterations to perform at each particle's timestep. Defaults to 20.
    tolerance (float): The preferred tolerance level. If this function does not reach the required tolerance level, the value with the lowest tolerance will be used. Defaults to 1e-3.

    Returns
    -------
    gca_v (NxMx3 numpy array): The guiding center trajectory for each particle.
    '''

    num_particles = np.shape(position)[0]
    steps         = np.shape(position)[1]

    gca_v = np.zeros(np.shape(position))

    for i in range(num_particles):
        for j in range(steps):
            B = b_field(position[i, j])
            GC_temp = position[i, j] - gyrovector(B, position[i, j], velocity[i, j], mass[i], charge[i])

            for k in range(max_iterations):
                B = b_field(GC_temp)
                GC = position[i, j] - gyrovector(B, GC_temp, velocity[i, j], mass[i], charge[i])
                if dot(GC - GC_temp, GC - GC_temp) / dot(GC, GC) < tolerance:
                    break
                GC_temp = GC

            gca_v[i, j] = GC
             
    return gca_v


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
