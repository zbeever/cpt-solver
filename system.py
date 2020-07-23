from multiprocessing import cpu_count, Pool
from scipy import constants as sp
import tqdm
import os
import h5py

from distributions import *
from fields import *
from integrators import *
from utils import *
from core import *

@njit
def solve_traj_numba(i, steps, dt, initial_conditions, particle_properties, integrator, drop_lost, downsample):
    hist_indiv    = np.zeros((steps, 4, 3))
    hist_indiv[0, :, :] = np.copy(initial_conditions[i, :, :])

    for j in range(steps - 1):
        hist_indiv[j + 1] = integrator(hist_indiv[j], particle_properties[i, :], dt, j)

        if drop_lost:
            if dot(hist_indiv[j + 1, 0, :], hist_indiv[j + 1, 0, :]) <= (Re + 100e3)**2:
                break

    return hist_indiv[::downsample, :, :]

class System:
    def __init__(self, e_field, b_field, integrator=relativistic_boris, drop_lost=False):
        self.e_field = e_field
        self.b_field = b_field

        self.drop_lost = drop_lost
        self.integrator = integrator(e_field, b_field)
        self.downsample = 1

        self.populated = False
        self.solved = False
        self.loaded = False

        return
    
    def populate(self, trials, r_dist, E_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e)):
        '''
        Populates the system by position and instantaneous pitch angle.

        Parameters
        ----------
        trials (int): The number of particles to add to the system.
        r_dist(): A 3D distribution function for the particles' positions (in m). This is obtained through the currying functions in distributions.py.
        e_dist(): A 1D distribution function for the particles' energies (in eV). This is obtained through the currying functions in distributions.py.
        pitch_ang_dist(): A 1D distribution function for the particles' pitch angles (in rad). This is obtained through the currying functions in distributions.py.
        phase_ang_dist(): A 1D distribution function for the particles' phase angles (in rad). This is obtained through the currying functions in distributions.py.
        m_dist(): A 1D distribution function for the particles' mass (in kg). This is obtained through the currying functions in distributions.py. Defaults to the electron mass.
        q_dist(): A 1D distribution function for the particles' charge (in C). This is obtained through the currying functions in distributions.py. Defaults to the electron charge.

        Returns
        -------
        None
        '''

        if self.loaded:
            raise NameError('Cannot populate an already solved system. Use add_particles.')

        self.n = trials

        self.initial_conditions  = np.zeros((self.n, 4, 3))
        self.particle_properties = np.zeros((self.n, 2))

        for i in tqdm.tqdm(range(self.n)):
            r = r_dist()
            K = E_dist()

            pitch_angle = pitch_ang_dist()
            phase_angle = phase_ang_dist()

            m = m_dist()
            q = q_dist()

            self.initial_conditions[i, 0] = r
            self.initial_conditions[i, 1] = velocity_vec(r, K, m, self.b_field, pitch_angle, phase_angle)
            self.initial_conditions[i, 2] = self.b_field(r)
            self.initial_conditions[i, 3] = self.e_field(r)

            self.particle_properties[i, 0] = m
            self.particle_properties[i, 1] = q
            
        self.populated = True

        return
    
    def populate_by_eq_pa(self, trials, L_dist, E_dist, eq_pitch_ang_dist, phase_ang_dist, m_dist = delta(sp.m_e), q_dist = delta(-sp.e), t = 0.):
        '''
        Populates the system by L-shell and equatorial pitch angle.

        TODO: At large equatorial pitch angles there seems to be a bias towards smaller initial values. Look into this.

        Parameters
        ----------
        trials (int): The number of particles to add to the system.
        L_dist(): A 1D distribution function for the particles' L-shells. This is obtained through the currying functions in distributions.py.
        E_dist(): A 1D distribution function for the particles' energies (in eV). This is obtained through the currying functions in distributions.py.
        eq_pitch_ang_dist(): A 1D distribution function for the particles' equatorial pitch angles (in rad). This is obtained through the currying functions in distributions.py.
        phase_ang_dist(): A 1D distribution function for the particles' phase angles (in rad). This is obtained through the currying functions in distributions.py.
        m_dist(): A 1D distribution function for the particles' mass (in kg). This is obtained through the currying functions in distributions.py. Defaults to the electron mass.
        q_dist(): A 1D distribution function for the particles' charge (in C). This is obtained through the currying functions in distributions.py. Defaults to the electron charge.

        Returns
        -------
        None
        '''

        if self.loaded:
            raise NameError('Cannot populate an already solved system. Use add_particles.')

        self.n = trials

        self.initial_conditions  = np.zeros((self.n, 4, 3))
        self.particle_properties = np.zeros((self.n, 2))

        for i in tqdm.tqdm(range(self.n)):
            # The field line of a given L-shell intersects the equatorial and meridional planes at -L_dist() * Re along the x-axis.
            mag_eq = np.array([-L_dist() * Re, 0, 0])

            # Get the array of points tracing the field line.
            rr = field_line(self.b_field, mag_eq, 1e-6)

            # Get the magnetic field along the field line.
            b_vec, b_mag, b_rad_mag = b_along_path(self.b_field, rr)

            # Isolate the magnetic field minimum (this coincides with the point of maximum curvature).
            b_min_ind = b_mag.argmin()
            b_min = b_mag[b_min_ind]

            K = E_dist()
            m = m_dist()
            q = q_dist()

            eq_pa = eq_pitch_ang_dist()

            # Get the magnetic field at the particle's mirror point.
            b_val = b_min / np.sin(eq_pa)**2

            # Find the point on the field line that has a magnetic field strength closest to the above calculated value.
            b_ind = np.abs(b_mag - b_val).argmin()

            # Get the sine of the near-mirror point pitch angle.
            b_ratio = np.sqrt(b_mag[b_ind] / b_min)
            new_sin = b_ratio * np.sin(eq_pa)

            # If this is greater than 1 (impossible), we take a step towards a less strong magnetic field and recalculate the near-mirror point pitch angle.
            if new_sin > 1:
                if b_ind > b_min_ind:
                    b_ind -= 1
                else:
                    b_ind += 1
                    
            b_ratio = np.sqrt(b_mag[b_ind] / b_min)
            new_sin = b_ratio * np.sin(eq_pa)

            # Get the spawn point of the particle.
            r = rr[b_ind]

            pitch_angle = np.arcsin(new_sin)
            phase_angle = phase_ang_dist()
            v = velocity_vec(r, K, m, self.b_field, pitch_angle, phase_angle)

            self.initial_conditions[i, 0] = r
            self.initial_conditions[i, 1] = v
            self.initial_conditions[i, 2] = self.b_field(r)
            self.initial_conditions[i, 3] = self.e_field(r)

            self.particle_properties[i, 0] = m
            self.particle_properties[i, 1] = q
            
        self.populated = True
        
        return

        
    def solve_traj(self, i):
        '''
        Wrapper for the single particle solver. Needed to use Numba.
        Avoid calling this directly. Instead, use System.solve(T, dt, sample_every).

        Parameters
        ----------
        i (int): The index of the particle to solve.

        Returns
        -------
        indiv_history (Nx4x3 numpy array): The history of the ith particle trajectory.
        '''

        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')

        if self.loaded:
            raise NameError('Cannot solve an already solved system. Use add_particles or add_time.')

        return solve_traj_numba(i, self.steps, self.dt, self.initial_conditions, self.particle_properties, self.integrator, self.drop_lost, self.downsample)

    
    def solve(self, T, dt, sample_every=1e-3):
        '''
        Solve the system. Only call after having populated the system.

        Parameters
        ----------
        T (float): The length of time to solve for.
        dt (float): The solver timestep. It's good practice to keep this under half the period of the fastest gyro-orbit.
        sample_every (float): The sample timestep.

        Returns
        -------
        indiv_history (Nx4x3 numpy array): The history of the ith particle trajectory.
        '''

        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')

        if self.loaded:
            raise NameError('Cannot solve an already solved system. Use add_particles or add_time.')
            
        self.steps = round(abs(T / dt))
        self.dt = dt
        
        # Our sample timestep must be greater than or equal to our solver timestep.
        if abs(sample_every) < dt:
            self.downsample = 1
        else:
            self.downsample = round(abs(sample_every) / abs(dt))
        
        # Divide each trajectory among the available processors.
        with Pool(cpu_count()) as p:
            results = list(tqdm.tqdm(p.imap(self.solve_traj, range(self.n)), total=self.n))

        self.history = np.array(results)
        self.solved = True

        return

    
    def write(self, filename):
        '''
        Writes the current system (the solved history) to an HDF5 file. If given the path of another file, the old file will be overwritten.

        Parameters
        ----------
        filename (string): The path to the file. Do not include the extension.

        Returns
        -------
        None
        '''

        if not self.solved or not self.populated:
            raise NameError('System not yet populated / solved. Cannot save empty IC / history array.')

        # Check to make sure the user actually wants to overwrite the specified file.
        if os.path.exists(f'{filename}.hdf5'):
            if input(f'{filename} already exists. Overwrite? (Y/N) ') != 'Y':
                print('Write canceled.')
                return
            else:
                os.remove(f'{filename}.hdf5')
            
        self.file = filename
        f = h5py.File(f'{self.file}.hdf5', 'a')
        f.create_group('data')

        data = f['data']
        hist = data.create_dataset('history', np.shape(self.history), maxshape=(None, None, None, None), dtype='float')
        pps  = data.create_dataset('particle_properties', np.shape(self.particle_properties), maxshape=(None, None), dtype='float')

        hist[:] = self.history
        pps[:]  = self.particle_properties

        data.attrs['dt'] = self.dt
        data.attrs['sample_dt'] = self.dt * self.downsample
        data.attrs['num_particles'] = self.n
        data.attrs['steps'] = self.steps
            
        f.close()

        file_size = os.path.getsize(f'{self.file}.hdf5')
        formatted_file_size = format_bytes(file_size)

        print(f'Saved file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')

        return
    

    def load(self, filename):
        '''
        Loads a file into the system. In reality, this only updates the system parameters and logs the given filename for use in other functions.

        Parameters
        ----------
        filename (string): The path to the file. Do not include the extension.

        Returns
        -------
        None
        '''

        if not os.path.exists(f'{filename}.hdf5'):
            raise NameError('No such file.')

        self.file = filename

        file_size = os.path.getsize(f'{self.file}.hdf5')
        formatted_file_size = format_bytes(file_size)

        print(f'Loaded file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')

        f = h5py.File(f'{self.file}.hdf5', 'a')

        data = f['data']
        hist = data['history']
        pps  = data['particle_properties']

        if hasattr(self, 'initial_conditions'):
            del self.initial_conditions
        self.initial_conditions = hist[:, -1, :, :]

        if hasattr(self, 'particle_properties'):
            del self.particle_properties
        self.particle_properties = pps[:, :]

        self.solved = True
        self.populated = True
        self.n = int(data.attrs['num_particles'])

        f.close()

        return


    def add_time(self, T):
        '''
        Given that a file has been created for / loaded into the system, continue the simulation for T seconds from where it left off.
        Automatically saves the results to the file upon completion.

        Parameters
        ----------
        T (float): The additional time for which to run the simulation.

        Returns
        -------
        None
        '''

        f = h5py.File(f'{self.file}.hdf5', 'a')

        data = f['data']
        hist = data['history']
        pps  = data['particle_properties']

        if hasattr(self, 'initial_conditions'):
            del self.initial_conditions
        self.initial_conditions = hist[:, -1, :, :]

        if hasattr(self, 'particle_properties'):
            del self.particle_properties
        self.particle_properties = pps[:, :]

        self.n = int(data.attrs['num_particles'])

        self.solve(T, float(data.attrs['dt']), float(data.attrs['sample_dt']))

        hist.resize((int(data.attrs['num_particles']), int(data.attrs['steps']) + np.shape(self.history)[1] - 1, 4, 3))
        hist[:, int(data.attrs['steps']):, :, :] = self.history[:, 1:, :, :]

        data.attrs['steps'] = hist.shape[1]
        
        f.close()


    def add_particles(self, trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e), by_eq_pa=False):
        '''
        Given that a file has been created for / loaded into the system, add additional trials to the simulation results.
        Automatically saves the results to the file upon completion.

        Parameters
        ----------
        trials (int): The number of particles to add to the system.
        L_dist(): A 1D distribution function for the particles' L-shells. This is obtained through the currying functions in distributions.py.
        E_dist(): A 1D distribution function for the particles' energies (in eV). This is obtained through the currying functions in distributions.py.
        eq_pitch_ang_dist(): A 1D distribution function for the particles' equatorial pitch angles (in rad). This is obtained through the currying functions in distributions.py.
        phase_ang_dist(): A 1D distribution function for the particles' phase angles (in rad). This is obtained through the currying functions in distributions.py.
        m_dist(): A 1D distribution function for the particles' mass (in kg). This is obtained through the currying functions in distributions.py. Defaults to the electron mass.
        q_dist(): A 1D distribution function for the particles' charge (in C). This is obtained through the currying functions in distributions.py. Defaults to the electron charge.

        Returns
        -------
        None
        '''

        if by_eq_pa:
            self.populate_by_eq_pa(trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e))
        else:
            self.populate(trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e))

        f = h5py.File(f'{self.file}.hdf5', 'a')

        data = f['data']
        hist = data['history']
        pps  = data['particle_properties']

        self.solve(float(data.attrs['sample_dt']) * float(data.attrs['steps']), float(data.attrs['dt']), float(data.attrs['sample_dt']))

        hist.resize((int(data.attrs['num_particles']) + np.shape(self.history)[0], int(data.attrs['steps']), 4, 3))
        hist[int(data.attrs['num_particles']):, :, :, :] = self.history[:, :, :, :]

        pps.resize((int(data.attrs['num_particles']) + np.shape(self.history)[0], 2))
        pps[int(data.attrs['num_particles']):, :] = self.particle_properties[:, :]

        data.attrs['num_particles'] = int(data.attrs['num_particles']) + trials
        
        f.close()


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
