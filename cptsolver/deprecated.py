from numba import njit, objmode
import numpy as np

from distributions import *
from utils import *
from fields import *
from integrators import *
from scipy import constants as sp
from math import sqrt


def solver(integrator):
    @njit
    def solve(history, intrinsic, dt):
        num_particles = np.shape(intrinsic)[0]
        steps = np.shape(history)[1]

        for i in range(num_particles):
            for j in range(steps - 1):
                history[i, j + 1] = integrator(history[i, j], intrinsic[i], dt, j)

    return solve


def populate(num_particles, steps, e_field, b_field, pos_dist, E_dist, pitch_angle_dist, phase_angle_dist, m_dist = delta(sp.m_e), q_dist = delta(-sp.e), t = 0.):
    history = np.zeros((num_particles, steps, 4, 3))
    intrinsic = np.zeros((num_particles, 2))

    for i in range(num_particles):
        r = pos_dist()
        K = E_dist()
        m = m_dist()
        q = q_dist()

        pitch_angle = pitch_angle_dist()
        phase_angle = phase_angle_dist()

        history[i, 0, 0] = r
        history[i, 0, 1] = velocity_vec(r, K, m, b_field, pitch_angle, phase_angle)
        history[i, 0, 2] = b_field(r, t)
        history[i, 0, 3] = e_field(r, t)
        intrinsic[i, 0] = m
        intrinsic[i, 1] = q

    return history, intrinsic


def populate_by_eq_pa(num_particles, steps, e_field, b_field, re_over_Re, E_dist, eq_pitch_angle_dist, phase_angle_dist, m_dist = delta(sp.m_e), q_dist = delta(-sp.e), t = 0.):
    history = np.zeros((num_particles, steps, 4, 3))
    intrinsic = np.zeros((num_particles, 2))

    for i in range(num_particles):
        mag_eq = np.array([-re_over_Re() * Re, 0, 0])
        rr = field_line(b_field, mag_eq, 1e1)

        K = E_dist()
        m = m_dist()
        q = q_dist()

        phase_angle = phase_angle_dist()

        r, pitch_angle = param_by_eq_pa(b_field, rr, eq_pitch_angle_dist())
        b = b_field(r)
        v = velocity_vec(r, K, m, b_field, pitch_angle, phase_angle)

        v_par = dot(b, v) / dot(b, b) * b
        v_perp = v - v_par
        gyrorad = gamma(v) * m * np.linalg.norm(v_perp) / (abs(q) * b)

        history[i, 0, 0] = r
        history[i, 0, 1] = v 
        history[i, 0, 2] = b_field(r, t)
        history[i, 0, 3] = e_field(r, t)
        intrinsic[i, 0] = m
        intrinsic[i, 1] = q

    return history, intrinsic

def plotter(field, history, intrinsic, dt, threshold=0.2, min_time=5e-3, padding=1e-3):
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 22})

    eq_pa_plots  = eq_pitch_angle_from_moment(history, intrinsic)
    eq_pa_values = get_eq_pas(field, history, intrinsic, threshold)
    pa           = pitch_angle(history)
    K            = kinetic_energy(history, intrinsic)
    v_pa, v_pam  = velocity_par(history)
    v_pe, v_pem  = velocity_perp(history)
    b            = b_mag(history) * 1e9
    r            = position(history)
    r_mag        = position_mag(history)
    gr           = gyrorad(history, intrinsic)
    gf           = gyrofreq(history, intrinsic)
    
    num_particles = len(history[:, 0, 0, 0])
    steps         = len(history[0, :, 0, 0])
    t_v          = np.arange(0, steps) * dt
    
    def plot(particle_ind):
        if type(particle_ind) != list:
            particle_ind = [particle_ind]
            
        n = len(particle_ind)
        
        fig = plt.figure(figsize=(20, 40))
        gs = GridSpec(10, 10, figure=fig)
        
        ax10 = fig.add_subplot(gs[0, 0:3])
        for i, j in enumerate(particle_ind):
            ax10.plot(r[j, :, 0] * inv_Re, r[j, :, 2] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax10.set_xlabel(r'$x_{GSM}$ ($R_E$)')
        ax10.set_ylabel(r'$z_{GSM}$ ($R_E$)')
        ax10.grid()
        
        ax11 = fig.add_subplot(gs[0, 3:6])
        for i, j in enumerate(particle_ind):
            ax11.plot(r[j, :, 0] * inv_Re, r[j, :, 1] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax11.set_xlabel(r'$x_{GSM}$ ($R_E$)')
        ax11.set_ylabel(r'$y_{GSM}$ ($R_E$)')
        ax11.grid()
        
        ax12 = fig.add_subplot(gs[0, 6:9])
        for i, j in enumerate(particle_ind):
            ax12.plot(r[j, :, 1] * inv_Re, r[j, :, 2] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax12.set_xlabel(r'$y_{GSM}$ ($R_E$)')
        ax12.set_ylabel(r'$z_{GSM}$ ($R_E$)')
        ax12.grid()
        
        ax1 = fig.add_subplot(gs[1, :])
        for i, j in enumerate(particle_ind):
            ax1.plot(t_v, eq_pa_plots[j, :], zorder=1, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
            for k in range(len(eq_pa_values[j, :, 0])):
                value   = eq_pa_values[j, k, 0]
                l_point = eq_pa_values[j, k, 1]
                r_point = eq_pa_values[j, k, 2]
                if value != -1.0:
                    ax1.hlines(value, l_point * dt, r_point * dt, zorder=2, linewidth=5, linestyle=':', colors=plt.cm.magma(i / n))
        ax1.set_xlim([0, steps * dt])
        ax1.set_ylim([0, 90])
        ax1.set_ylabel('Equatorial\nPitch angle (deg)')
        ax1.grid()
        
        ax2 = fig.add_subplot(gs[2, :])
        for i, j in enumerate(particle_ind):
            ax2.plot(t_v, pa[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax2.set_xlim([0, steps * dt])
        ax2.set_ylim([0, 180])
        ax2.set_ylabel('Pitch angle (deg)')
        ax2.grid()
        
        ax3 = fig.add_subplot(gs[3, :])
        for i, j in enumerate(particle_ind):
            ax3.plot(t_v, r_mag[j, :] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax3.set_xlim([0, steps * dt])
        ax3.set_ylim([0, np.amax(r_mag[particle_ind, :]) * inv_Re])
        ax3.set_ylabel('Distance from\nGSM origin ($R_E$)')
        ax3.grid()
        
        ax4 = fig.add_subplot(gs[4, :])
        for i, j in enumerate(particle_ind):
            ax4.plot(t_v, K[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax4.set_xlim([0, steps * dt])
        ax4.set_ylim([np.amin(K[particle_ind, 0]) * (1 - 1e-3), np.amax(K[particle_ind, 0]) * (1 + 1e-3)])
        ax4.set_ylabel('Energy (eV)')
        ax4.grid()
        
        ax5 = fig.add_subplot(gs[5, :])
        for i, j in enumerate(particle_ind):
            ax5.plot(t_v, b[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax5.set_xlim([0, steps * dt])
        ax5.set_ylim([np.amin(b[particle_ind, :]) * (1 - 1e-1), np.amax(b[particle_ind, :]) * (1 + 1e-1)])
        ax5.set_ylabel(r'$\|B\|$ (nT)')
        ax5.grid()
        
        ax6 = fig.add_subplot(gs[6, :])
        for i, j in enumerate(particle_ind):
            ax6.plot(t_v, v_pam[j, :] / sp.c, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax6.set_xlim([0, steps * dt])
        ax6.set_ylim([0, 1])
        ax6.set_ylabel(r'$v_{\parallel}/c$')
        ax6.grid()
        
        ax7 = fig.add_subplot(gs[7, :])
        for i, j in enumerate(particle_ind):
            ax7.plot(t_v, v_pem[j, :] / sp.c, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax7.set_xlim([0, steps * dt])
        ax7.set_ylim([0, 1])
        ax7.set_ylabel(r'$v_{\perp}/c$')
        ax7.grid()
        
        ax8 = fig.add_subplot(gs[8, :])
        for i, j in enumerate(particle_ind):
            ax8.plot(t_v, gr[j, :] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax8.set_xlim([0, steps * dt])
        ax8.set_ylim([0, np.amax(gr[particle_ind, :]) * inv_Re])
        ax8.set_ylabel(r'Gyroradius ($R_E$)')
        ax8.grid()
        
        ax9 = fig.add_subplot(gs[9, :])
        for i, j in enumerate(particle_ind):
            ax9.plot(t_v, gf[j, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        ax9.set_xlim([0, steps * dt])
        ax9.set_ylim([0, np.amax(gf[particle_ind, :])])
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel(r'Gyrofrequency (s$^{-1}$)')
        ax9.grid()
        
        fig.tight_layout(pad=0.4)
        handles, labels = ax9.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.show()
        
    return plot

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

t = 0


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
            if (velocity[i, j] == 0).all() and (b_field[i, j] == 0).all():
                break

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
            if b_magnitude[i, j] == 0 and ke[i, j] == 0 and v_perp_magnitude[i, j] == 0:
                break

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
            if ke[i, j] == 0 and b_magnitude[i, j] == 0:
                break

            gamma_v = ke[i, j] / J_to_eV(mass[i] * sp.c**2) + 1
            gyrofreq_v[i, j] = abs(charge[i]) * b_magnitude[i, j] / (2 * np.pi * gamma_v * mass[i])

    return gyrofreq_v


def eq_pitch_angle(b_field, pitch_ang, b_magnitude, gyrorad, position, unwrapped=False):
    '''
    Calculates the equatorial pitch angle (in radians) along a history. Does so by taking advantage of the fact that the first adiabatic invariant
    is most accurate at the mirror points along a particle's trajecotry (see Mozer (1966) DOI: 10.1029/JZ071i011p02701 and Anderson (1997) DOI:
    10.1029/97JA00798). A single equatorial pitch angle is assigned to the interval between adjacent equatorial crossing points.

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

        if len(mp_ind) == 0:
            continue
        
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
            if (position[i, j] == 0).all() and (velocity[i, j] == 0).all():
                break

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


def eq_pitch_angle_ind(i, b_field, pitch_ang, b_magnitude, gyrorad, position, time, unwrapped=False):
    '''
    Calculates the equatorial pitch angle (in radians) along a trajectory. Does so by taking advantage of the fact that the first adiabatic invariant
    is most accurate at the mirror points along a particle's trajecotry (see Mozer (1966) DOI: 10.1029/JZ071i011p02701 and Anderson (1997) DOI:
    10.1029/97JA00798). A single equatorial pitch angle is assigned to the interval between adjacent equatorial crossing points.

    Parameters
    ----------
    i (int): The index of the particle to use.
    b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py). This is required for checking adiabaticity.
    pitch_ang (NxM numpy array): A history of particle pitch angles. The first index denotes the particle and the second the timestep.
    b_magnitude (NxM numpy array): A history of background magnetic field strengths. The first index denotes the particle and the second the timestep.
    gyrorad (NxM numpy array): A history of gyroradii. The first index denotes the particle and the second the timestep.
    position (NxMx3 numpy array): A history of particle locations. The first index denotes the particle, the second the timestep, and the third the dimension.
    unwrapped (bool): Whether the equatorial pitch angle should be displayed from 0 to pi / 2 or unwrapped and displayed from 0 to pi.

    Returns
    -------
    eq_pa_ind_v (NxM numpy array): The equatorial pitch angle (in radians) at each timestep for the particle.
    '''

    steps = np.shape(pitch_ang)[1]
    eq_pa_ind_v = np.zeros(steps)
    
    eps = 2e-1

    # Find mirror points (where the particle has a pitch angle of 90 degrees).
    mp_ind = np.argwhere(np.diff(np.sign(pitch_ang[i, :] - np.pi / 2)) != 0)[:, 0]

    if len(mp_ind) == 0:
        return eq_pa_ind_v
    
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
        R_c = flc(b_field, position[i, 0], time[0])
        if gyrorad[i, 0] / R_c < 1e-3:
            mp_ind = np.append(0, mp_ind)
    if x_ind_fixed[-1] > mp_ind[-1]:
        R_c = flc(b_field, position[i, -1], time[-1])
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
            eq_pa_ind_v[display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1] = np.pi / 2
        elif unwrapped is True:
            eq_pa_ind_v[display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1] = np.mod(np.sign(np.pi / 2 - pitch_ang[i, display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1]) * np.arcsin(eq_pa_sin), np.pi)
        else:
            eq_pa_ind_v[display_xs[k]:display_xs[min(k + 1, len(display_xs) - 1)] + 1] = np.arcsin(eq_pa_sin)
            
    return eq_pa_ind_v


def plot(self, particle_ind, threshold=0.1):
    if not self.solved:
        raise NameError('System not yet solved. Cannot plot empty history array.')
        
    if type(particle_ind) != list:
        particle_ind = [particle_ind]
        
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 22})

    eq_pa_plots  = eq_pitch_angle_from_moment(self.history[particle_ind, :, :], self.ics[particle_ind, 4, 0:2])
    eq_pa_values = get_eq_pas(self.b_field, self.history[particle_ind, :, :], self.ics[particle_ind, 4, 0:2], threshold)
    pa           = pitch_angle(self.history[particle_ind, :, :])
    K            = kinetic_energy(self.history[particle_ind, :, :], self.ics[particle_ind, 4, 0:2])
    v_pa, v_pam  = velocity_par(self.history[particle_ind, :, :])
    v_pe, v_pem  = velocity_perp(self.history[particle_ind, :, :])
    b            = b_mag(self.history[particle_ind, :, :]) * 1e9
    r            = position(self.history[particle_ind, :, :])
    r_mag        = position_mag(self.history[particle_ind, :, :])
    gr           = gyrorad(self.history[particle_ind, :, :], self.ics[particle_ind, 4, 0:2])
    gf           = gyrofreq(self.history[particle_ind, :, :], self.ics[particle_ind, 4, 0:2])

    num_particles = len(particle_ind)
    steps         = len(self.history[0, :, 0, 0])
    t_v           = np.arange(0, steps) * self.dt
        
    n = len(particle_ind)
    
    fig = plt.figure(figsize=(20, 40))
    gs = GridSpec(10, 10, figure=fig)
    
    ax10 = fig.add_subplot(gs[0, 0:3])
    for i, j in enumerate(particle_ind):
        ax10.plot(r[i, :, 0] * inv_Re, r[i, :, 2] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax10.set_xlabel(r'$x_{GSM}$ ($R_E$)')
    ax10.set_ylabel(r'$z_{GSM}$ ($R_E$)')
    ax10.grid()
    
    ax11 = fig.add_subplot(gs[0, 3:6])
    for i, j in enumerate(particle_ind):
        ax11.plot(r[i, :, 0] * inv_Re, r[i, :, 1] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax11.set_xlabel(r'$x_{GSM}$ ($R_E$)')
    ax11.set_ylabel(r'$y_{GSM}$ ($R_E$)')
    ax11.grid()
    
    ax12 = fig.add_subplot(gs[0, 6:9])
    for i, j in enumerate(particle_ind):
        ax12.plot(r[i, :, 1] * inv_Re, r[i, :, 2] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax12.set_xlabel(r'$y_{GSM}$ ($R_E$)')
    ax12.set_ylabel(r'$z_{GSM}$ ($R_E$)')
    ax12.grid()
    
    ax1 = fig.add_subplot(gs[1, :])
    for i, j in enumerate(particle_ind):
        ax1.plot(t_v, eq_pa_plots[i, :], zorder=1, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
        for k in range(len(eq_pa_values[i, :, 0])):
            value   = eq_pa_values[i, k, 0]
            l_point = eq_pa_values[i, k, 1]
            r_point = eq_pa_values[i, k, 2]
            if value != -1.0:
                ax1.hlines(value, l_point * self.dt, r_point * self.dt, zorder=2, linewidth=5, linestyle=':', colors=plt.cm.magma(i / n))
    ax1.set_xlim([0, steps * self.dt])
    ax1.set_ylim([0, 90])
    ax1.set_ylabel('Equatorial\nPitch angle (deg)')
    ax1.grid()
    
    ax2 = fig.add_subplot(gs[2, :])
    for i, j in enumerate(particle_ind):
        ax2.plot(t_v, pa[i, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax2.set_xlim([0, steps * self.dt])
    ax2.set_ylim([0, 180])
    ax2.set_ylabel('Pitch angle (deg)')
    ax2.grid()
    
    ax3 = fig.add_subplot(gs[3, :])
    for i, j in enumerate(particle_ind):
        ax3.plot(t_v, r_mag[i, :] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax3.set_xlim([0, steps * self.dt])
    ax3.set_ylim([0, np.amax(r_mag[:, :]) * inv_Re])
    ax3.set_ylabel('Distance from\nGSM origin ($R_E$)')
    ax3.grid()
    
    ax4 = fig.add_subplot(gs[4, :])
    for i, j in enumerate(particle_ind):
        ax4.plot(t_v, K[i, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax4.set_xlim([0, steps * self.dt])
    ax4.set_ylim([np.amin(K[:, 0]) * (1 - 1e-3), np.amax(K[:, 0]) * (1 + 1e-3)])
    ax4.set_ylabel('Energy (eV)')
    ax4.grid()
    
    ax5 = fig.add_subplot(gs[5, :])
    for i, j in enumerate(particle_ind):
        ax5.plot(t_v, b[i, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax5.set_xlim([0, steps * self.dt])
    ax5.set_ylim([np.amin(b[:, :]) * (1 - 1e-1), np.amax(b[:, :]) * (1 + 1e-1)])
    ax5.set_ylabel(r'$\|B\|$ (nT)')
    ax5.grid()
    
    ax6 = fig.add_subplot(gs[6, :])
    for i, j in enumerate(particle_ind):
        ax6.plot(t_v, v_pam[i, :] / sp.c, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax6.set_xlim([0, steps * self.dt])
    ax6.set_ylim([0, 1])
    ax6.set_ylabel(r'$v_{\parallel}/c$')
    ax6.grid()
    
    ax7 = fig.add_subplot(gs[7, :])
    for i, j in enumerate(particle_ind):
        ax7.plot(t_v, v_pem[i, :] / sp.c, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax7.set_xlim([0, steps * self.dt])
    ax7.set_ylim([0, 1])
    ax7.set_ylabel(r'$v_{\perp}/c$')
    ax7.grid()
    
    ax8 = fig.add_subplot(gs[8, :])
    for i, j in enumerate(particle_ind):
        ax8.plot(t_v, gr[i, :] * inv_Re, c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax8.set_xlim([0, steps * self.dt])
    ax8.set_ylim([0, np.amax(gr[:, :]) * inv_Re])
    ax8.set_ylabel(r'Gyroradius ($R_E$)')
    ax8.grid()
    
    ax9 = fig.add_subplot(gs[9, :])
    for i, j in enumerate(particle_ind):
        ax9.plot(t_v, gf[i, :], c=plt.cm.plasma(i / n), label=chr(ord('@') + i + 1))
    ax9.set_xlim([0, steps * self.dt])
    ax9.set_ylim([0, np.amax(gf[:, :])])
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel(r'Gyrofrequency (s$^{-1}$)')
    ax9.grid()
    
    fig.tight_layout(pad=0.4)
    handles, labels = ax9.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()
    
    return
