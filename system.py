from multiprocessing import cpu_count, Pool
from scipy import constants as sp
import tqdm

from distributions import *
from fields import *
from integrators import *
from utils import *
from core import *
from field_utils import *

class System:
    def __init__(self, e_field, b_field):
        self.e_field = e_field
        self.b_field = b_field
        self.integrator = relativistic_boris(e_field, b_field)
        self.downsample = 1
        self.populated = False
        self.solved = False
        return
    
    def populate(self, trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e)):
        self.n   = trials
        self.ics = np.zeros((self.n, 5, 3))

        for i in tqdm.tqdm(range(self.n)):
            r = r_dist()
            K = e_dist()
            pitch_angle = pitch_ang_dist()
            phase_angle = phase_ang_dist()
            m = m_dist()
            q = q_dist()

            self.ics[i, 0]    = r
            self.ics[i, 1]    = velocity_vec(r, K, m, self.b_field, pitch_angle, phase_angle)
            self.ics[i, 2]    = self.b_field(r)
            self.ics[i, 3]    = self.e_field(r)
            self.ics[i, 4, 0] = m
            self.ics[i, 4, 1] = q
            
        self.populated = True

        return
    
    def populate_by_eq_pa(self, trials, re_over_Re_dist, E_dist, eq_pitch_ang_dist, phase_ang_dist, m_dist = delta(sp.m_e), q_dist = delta(-sp.e), t = 0.):
        self.n   = trials
        self.ics = np.zeros((self.n, 5, 3))

        for i in tqdm.tqdm(range(self.n)):
            mag_eq = np.array([-re_over_Re_dist() * Re, 0, 0])
            rr = field_line(self.b_field, mag_eq, 1e1)

            K = E_dist()
            m = m_dist()
            q = q_dist()

            phase_angle = phase_ang_dist()

            r, pitch_angle = param_by_eq_pa(self.b_field, rr, eq_pitch_ang_dist())
            v = velocity_vec(r, K, m, self.b_field, pitch_angle, phase_angle)

            self.ics[i, 0]    = r
            self.ics[i, 1]    = v
            self.ics[i, 2]    = self.b_field(r)
            self.ics[i, 3]    = self.e_field(r)
            self.ics[i, 4, 0] = m
            self.ics[i, 4, 1] = q
            
            
        self.populated = True
        
        return
        
    def solve_traj(self, i):
        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')
            
        hist_indiv    = np.zeros((self.steps, 4, 3))
        hist_indiv[0] = np.copy(self.ics[i, 0:4, :])

        for j in range(self.steps - 1):
            hist_indiv[j + 1] = self.integrator(hist_indiv[j], self.ics[i, 4, 0:2], self.ics[i, 4, 2], j)

        return hist_indiv[::self.downsample, :, :]
    
    def solve(self, T, dt, sample_every=1e-3):
        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')
            
        self.steps = round(abs(T / dt))
        self.dt = self.ics[:, 4, 2] = dt
        
        if abs(sample_every) < dt:
            self.downsample = 1
        else:
            self.downsample = round(abs(sample_every) / abs(dt))
        
        with Pool(cpu_count()) as p:
            results = list(tqdm.tqdm(p.imap(self.solve_traj, range(self.n)), total=self.n))

        self.history = np.array(results)
        self.solved = True
        
        return
    
    def save(self, file_name):
        if not self.solved or not self.populated:
            raise NameError('System not yet populated / solved. Cannot save empty IC / history array.')
            
        np.save(f'simulations/{file_name}_hist.npy', self.history)
        np.save(f'simulations/{file_name}_ics.npy', self.ics)
        print(f'Saved files containing {format_bytes(self.history.nbytes + self.ics.nbytes)[0]:.2f} {format_bytes(self.history.nbytes + self.ics.nbytes)[1]} of information.')
        return
    
    def load(self, file_name):
        self.history = np.load(f'simulations/{file_name}_hist.npy')
        self.ics = np.load(f'simulations/{file_name}_ics.npy')
        self.dt = self.ics[0, 4, 2]
        print(f'Loaded files containing {format_bytes(self.history.nbytes + self.ics.nbytes)[0]:.2f} {format_bytes(self.history.nbytes + self.ics.nbytes)[1]} of information.')
        self.populated = True
        self.solved = True
        return
    
    def decimate(self, ds):
        if not self.solved:
            raise NameError('System not yet solved. Cannot decimate empty history array.')
        
        size_before = self.history.nbytes
        self.history = self.history[:, ::ds, :, :]
        size_after = self.history.nbytes        
        print(f'Removed {format_bytes(size_before - size_after)[0]:.2f} {format_bytes(size_before - size_after)[1]} of information.')
        return
    
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
        t_v          = np.arange(0, steps) * self.dt
            
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
