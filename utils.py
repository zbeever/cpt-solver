import numpy as np
from numba import njit
from diagnostics import *
import scipy.constants as sp
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt

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


def plotter(history, intrinsic, dt):
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 22})

    eq_pa_plots  = eq_pitch_angle_from_moment(history, intrinsic)
    eq_pa_values = get_eq_pas(history, intrinsic, dt)
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
        
        fig = plt.figure(figsize=(20, 30))
        gs = GridSpec(8, 6, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, :])
        for i, j in enumerate(particle_ind):
            ax1.plot(t_v, eq_pa_plots[j, :], zorder=1, c=plt.cm.plasma(i / n))
            for k in range(len(eq_pa_values[j, :, 0])):
                value   = eq_pa_values[j, k, 0]
                l_point = eq_pa_values[j, k, 1]
                r_point = eq_pa_values[j, k, 2]
                if value != -1.0:
                    ax1.hlines(value, l_point * dt, r_point * dt, zorder=2, linewidth=5, linestyle=':', colors=plt.cm.magma(i / n))
        ax1.set_xlim([0, steps * dt])
        ax1.set_ylim([0, 90])
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Equatorial\nPitch angle (deg)')
        ax1.grid()
        
        ax2 = fig.add_subplot(gs[1, :])
        for i, j in enumerate(particle_ind):
            ax2.plot(t_v, pa[j, :], c=plt.cm.plasma(i / n))
        ax2.set_xlim([0, steps * dt])
        ax2.set_ylim([0, 90])
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pitch angle (deg)')
        ax2.grid()
        
        ax3 = fig.add_subplot(gs[2, :])
        for i, j in enumerate(particle_ind):
            ax3.plot(t_v, r_mag[j, :] * inv_Re, c=plt.cm.plasma(i / n))
        ax3.set_xlim([0, steps * dt])
        ax3.set_ylim([0, np.amax(r_mag[particle_ind, :]) * inv_Re])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Distance from\nGSM origin ($R_E$)')
        ax3.grid()
        
        ax4 = fig.add_subplot(gs[3, :])
        for i, j in enumerate(particle_ind):
            ax4.plot(t_v, K[j, :], c=plt.cm.plasma(i / n))
        ax4.set_xlim([0, steps * dt])
        ax4.set_ylim([np.amin(K[particle_ind, 0]) * (1 - 1e-3), np.amax(K[particle_ind, 0]) * (1 + 1e-3)])
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Energy (eV)')
        ax4.grid()
        
        ax5 = fig.add_subplot(gs[4, :])
        for i, j in enumerate(particle_ind):
            ax5.plot(t_v, b[j, :], c=plt.cm.plasma(i / n))
        ax5.set_xlim([0, steps * dt])
        ax5.set_ylim([np.amin(b[particle_ind, :]) * (1 - 1e-1), np.amax(b[particle_ind, :]) * (1 + 1e-1)])
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel(r'$\|B\|$ (nT)')
        ax5.grid()
        
        ax6 = fig.add_subplot(gs[5, 0:3])
        for i, j in enumerate(particle_ind):
            ax6.plot(t_v, v_pam[j, :] / sp.c, c=plt.cm.plasma(i / n))
        ax6.set_xlim([0, steps * dt])
        ax6.set_ylim([0, 1])
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel(r'$v_{\parallel}/c$')
        ax6.grid()
        
        ax7 = fig.add_subplot(gs[5, 3:6])
        for i, j in enumerate(particle_ind):
            ax7.plot(t_v, v_pem[j, :] / sp.c, c=plt.cm.plasma(i / n))
        ax7.set_xlim([0, steps * dt])
        ax7.set_ylim([0, 1])
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel(r'$v_{\perp}/c$')
        ax7.grid()
        
        ax8 = fig.add_subplot(gs[6, 0:3])
        for i, j in enumerate(particle_ind):
            ax8.plot(t_v, gr[j, :] * inv_Re, c=plt.cm.plasma(i / n))
        ax8.set_xlim([0, steps * dt])
        ax8.set_ylim([0, np.amax(gr[particle_ind, :]) * inv_Re])
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel(r'Gyroradius ($R_E$)')
        ax8.grid()
        
        ax9 = fig.add_subplot(gs[6, 3:6])
        for i, j in enumerate(particle_ind):
            ax9.plot(t_v, gf[j, :], c=plt.cm.plasma(i / n))
        ax9.set_xlim([0, steps * dt])
        ax9.set_ylim([0, np.amax(gf[particle_ind, :])])
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel(r'Gyrofrequency (s$^{-1}$)')
        ax9.grid()
        
        ax10 = fig.add_subplot(gs[7, 0:2])
        for i, j in enumerate(particle_ind):
            ax10.plot(r[j, :, 0] * inv_Re, r[j, :, 2] * inv_Re, c=plt.cm.plasma(i / n))
        ax10.set_xlabel(r'$x_{GSM}$ ($R_E$)')
        ax10.set_ylabel(r'$z_{GSM}$ ($R_E$)')
        ax10.grid()
        
        ax11 = fig.add_subplot(gs[7, 2:4])
        for i, j in enumerate(particle_ind):
            ax11.plot(r[j, :, 0] * inv_Re, r[j, :, 1] * inv_Re, c=plt.cm.plasma(i / n))
        ax11.set_xlabel(r'$x_{GSM}$ ($R_E$)')
        ax11.set_ylabel(r'$y_{GSM}$ ($R_E$)')
        ax11.grid()
        
        ax12 = fig.add_subplot(gs[7, 4:6])
        for i, j in enumerate(particle_ind):
            ax12.plot(r[j, :, 1] * inv_Re, r[j, :, 2] * inv_Re, c=plt.cm.plasma(i / n))
        ax12.set_xlabel(r'$y_{GSM}$ ($R_E$)')
        ax12.set_ylabel(r'$z_{GSM}$ ($R_E$)')
        ax12.grid()
        
        fig.tight_layout(pad=1.0)
        plt.show()
        
    return plot
