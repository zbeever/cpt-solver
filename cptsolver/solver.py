import os
from multiprocessing import cpu_count, Pool

import tqdm
import h5py
import numpy as np
import scipy.constants as sp
from numba import njit, prange

from cptsolver.integrators import relativistic_boris
from cptsolver.distributions import delta
from cptsolver.utils import Re, field_line, b_along_path, velocity_vec, solve_traj, format_bytes, solve_sys


class solver:
    '''
    Used to save, load, solve, and extend systems of particles in magnetic and electric fields.

    Methods
    -------
    populate(self, trials, r_dist, E_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e))
       Populates a system of particles following the specified distributions. 

    populate_by_eq_pa(self, trials, L, E_dist, eq_pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e), t=0.)
        Populates a system of particles along a particular L-shell following the specified distributions.

    solve_traj(self, i)
        Solves the trajectory of a single particle (indexed by i).

    solve(self, T, dt, sample_every=1e-3)
        Solves a system of particles.

    add_time(self, T)
        Extends an already-solved system by T seconds.

    add_particles(self, trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e), by_eq_pa=False)
        Extends an already-solved system with N particles where N = trials. Populates these following the specified distributions.

    save(self, filename)
        Saves the system to a file for use with the analyzer.

    load(self, filename)
        Loads a system from a file for the purpose of extending it.

    '''

    def __init__(self, e_field, b_field, integrator=relativistic_boris, drop_lost=False):
        '''
        Associates background fields and an integrator with the system.

        Parameters
        ----------
        e_field(r, t=0.) : function
            The electric field function (this is obtained through the currying functions in fields.py). Accepts a
            position (float[3]) and time (float). Returns the electric field vector (float[3]) at that point in spacetime.

        b_field(r, t=0.) : function
            The magnetic field function (this is obtained through the currying functions in fields.py). Accepts a
            position (float[3]) and time (float). Returns the magnetic field vector (float[3]) at that point in spacetime.

        integrator(e_field, b_field) : function, optional
            The particle pusher to use. Accepts an electric (function) and magnetic (function) field. Returns step(state, particle_properties, dt, step_num) (function).
            Defaults to the relativistic Boris integrator.

        drop_lost : bool, optional
            Whether particles lost to the atmosphere should be dropped from the simulation.
        '''

        self.e_field = e_field
        self.b_field = b_field

        self.drop_lost = drop_lost
        self.integrator = integrator(e_field, b_field)
        self.downsample = 1

        self.populated = False
        self.solved = False
        self.loaded = False

        return

    
    def save(self, filename):
        '''
        Writes the current system (the solved history) to an HDF5 file. If given the path of another file, the old file will be overwritten.

        Parameters
        ----------
        filename : string
            The path to the file. Do not include the extension.

        Returns
        -------
        None
        '''

        # Disallow the user to save an empty file
        if not self.solved or not self.populated:
            raise NameError('System not yet populated / solved. Cannot save empty IC / history array.')

        # Check to make sure the user actually wants to overwrite the specified file
        if os.path.exists(f'{filename}.hdf5'):
            if input(f'{filename} already exists. Overwrite? (Y/N) ') != 'Y':
                print('Write canceled.')
                return
            else:
                os.remove(f'{filename}.hdf5')
            
        # Create a new .hdf5 file and access the main group (history)
        f = h5py.File(f'{filename}.hdf5', 'a')
        f.create_group('history')
        hist = f['history']

        # Log the filename for use in other functions
        self.file = filename

        # Create the datasets for the solver's data
        rr = hist.create_dataset('position',       np.shape(self.history[:, :, 0, :]),       maxshape=(None, None, None), dtype='float', compression='gzip')
        vv = hist.create_dataset('velocity',       np.shape(self.history[:, :, 1, :]),       maxshape=(None, None, None), dtype='float', compression='gzip')
        bb = hist.create_dataset('magnetic_field', np.shape(self.history[:, :, 2, :]),       maxshape=(None, None, None), dtype='float', compression='gzip')
        ee = hist.create_dataset('electric_field', np.shape(self.history[:, :, 3, :]),       maxshape=(None, None, None), dtype='float', compression='gzip')
        mm = hist.create_dataset('mass',           np.shape(self.particle_properties[:, 0]), maxshape=(None, ),           dtype='float', compression='gzip')
        qq = hist.create_dataset('charge',         np.shape(self.particle_properties[:, 1]), maxshape=(None, ),           dtype='float', compression='gzip')
        tt = hist.create_dataset('time',           np.shape(self.history[0, :, 0, 0]),       maxshape=(None, ),           dtype='float', compression='gzip')

        # Set the solved system's attributes
        hist.attrs['solve_timestep']  = self.dt
        hist.attrs['sample_timestep'] = self.dt * self.downsample
        hist.attrs['num_particles']   = self.n
        hist.attrs['steps']           = np.shape(self.history)[1]
        hist.attrs['drop_lost']       = self.drop_lost

        # Copy the solved system to the file
        rr[:] = self.history[:, :, 0, :]
        vv[:] = self.history[:, :, 1, :]
        bb[:] = self.history[:, :, 2, :]
        ee[:] = self.history[:, :, 3, :]
        mm[:] = self.particle_properties[:, 0]
        qq[:] = self.particle_properties[:, 1]
        tt[:] = np.arange(0, int(hist.attrs['steps'])) * hist.attrs['sample_timestep']
            
        # Close the file and mark the solver as having loaded the solved system (to prevent an accidental re-solve)
        f.close()
        self.loaded = True

        # Get the saved file's size and let the user know the process has been completed
        file_size = os.path.getsize(f'{self.file}.hdf5')
        formatted_file_size = format_bytes(file_size)
        print(f'Saved file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')

        return
    

    def load(self, filename):
        '''
        Loads a file into the system. In reality, this only updates the system parameters and logs the given filename for use in other functions.

        Parameters
        ----------
        filename : string
            The path to the file. Do not include the extension.

        Returns
        -------
        None
        '''

        # We cannot load a file that does not exist
        if not os.path.exists(f'{filename}.hdf5'):
            raise NameError('No such file.')

        # Open the .hdf5 file and access the main group (history)
        f = h5py.File(f'{filename}.hdf5', 'a')
        hist = f['history']

        # Log the filename and the number of particles for use in other functions
        self.file = filename
        self.n = int(hist.attrs['num_particles'])

        # Close the file and mark the system as solved and populated (to prevent an accidental re-solve)
        f.close()
        self.solved = True
        self.populated = True

        # Get the saved file's size and let the user know the process has been completed
        file_size = os.path.getsize(f'{filename}.hdf5')
        formatted_file_size = format_bytes(file_size)
        print(f'Loaded file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')

        return


    def populate(self, trials, r_dist, E_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e)):
        '''
        Populates the system by position and instantaneous pitch angle.

        Parameters
        ----------
        trials : int
            The number of particles to add to the system.

        r_dist() : function
            A 3D distribution function for the particles' positions (in m). This is obtained through the currying functions in distributions.py.

        e_dist() : function
            A 1D distribution function for the particles' energies (in eV). This is obtained through the currying functions in distributions.py.

        pitch_ang_dist() : function
            A 1D distribution function for the particles' pitch angles (in rad). This is obtained through the currying functions in distributions.py.

        phase_ang_dist() : function A 1D distribution function for the particles' phase angles (in rad). This is obtained through the currying functions in distributions.py.

        m_dist() : function, optional
            A 1D distribution function for the particles' mass (in kg). This is obtained through the currying functions in distributions.py. Defaults to the electron mass.

        q_dist() : function, optional
            A 1D distribution function for the particles' charge (in C). This is obtained through the currying functions in distributions.py. Defaults to the electron charge.

        Returns
        -------
        None
        '''

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
        self.solved = False

        return

    
    def populate_by_eq_pa(self, trials, L, E_dist, eq_pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e), t=0., planar=False):
        '''
        Populates the system by L-shell and equatorial pitch angle.

        Parameters
        ----------
        trials : int
            The number of particles to add to the system.

        L : float
            The L-shell at which to spawn particles.

        r_dist() : function
            A 3D distribution function for the particles' positions (in m). This is obtained through the currying functions in distributions.py.

        e_dist() : function
            A 1D distribution function for the particles' energies (in eV). This is obtained through the currying functions in distributions.py.

        pitch_ang_dist() : function
            A 1D distribution function for the particles' pitch angles (in rad). This is obtained through the currying functions in distributions.py.

        phase_ang_dist() : function A 1D distribution function for the particles' phase angles (in rad). This is obtained through the currying functions in distributions.py.

        m_dist() : function, optional
            A 1D distribution function for the particles' mass (in kg). This is obtained through the currying functions in distributions.py. Defaults to the electron mass.

        q_dist() : function, optional
            A 1D distribution function for the particles' charge (in C). This is obtained through the currying functions in distributions.py. Defaults to the electron charge.

        Returns
        -------
        None
        '''

        self.n = trials

        self.initial_conditions  = np.zeros((self.n, 4, 3))
        self.particle_properties = np.zeros((self.n, 2))

        # The field line of a given L-shell intersects the equatorial and meridional planes at -L_dist() * Re along the x-axis.
        mag_eq = np.array([-L * Re, 0, 0])

        # Get the array of points tracing the field line.
        rr = field_line(self.b_field, mag_eq, planar=planar)

        # Get the magnetic field along the field line.
        b_vec, b_mag, b_rad_mag = b_along_path(self.b_field, rr)

        # Isolate the magnetic field minimum (this coincides with the point of maximum curvature).
        b_min_ind = b_mag.argmin()
        b_min = b_mag[b_min_ind]

        for i in tqdm.tqdm(range(self.n)):
            K = E_dist()
            m = m_dist()
            q = q_dist()

            eq_pa = eq_pitch_ang_dist()

            # Get the magnetic field at the particle's mirror point.
            b_val = b_min / np.sin(eq_pa)**2

            # Find the point on the field line that has a magnetic field strength closest to the above calculated value.
            b_ind = 0

            for j, b in enumerate(b_mag):
                if b > b_val:
                    continue
                else:
                    b_ind = j
                    break

            # b_ind = np.abs(b_mag - b_val).argmin()

            # Get the sine of the near-mirror point pitch angle.
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
        self.solved = False
        self.loaded = False
        
        return

        
    def solve_traj(self, i):
        '''
        Wrapper for the single particle solver. Needed to use Numba. Avoid calling this directly.

        Parameters
        ----------
        i : int
            The index of the particle to solve.

        Returns
        -------
        indiv_history : float[Nx4x3]
            The history of the ith particle trajectory.
        '''

        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')

        if self.loaded:
            raise NameError('Cannot solve an already solved system. Use add_particles or add_time.')

        return solve_traj(i, self.steps, self.dt, self.initial_conditions, self.particle_properties, self.integrator, self.drop_lost, self.downsample)

    
    def solve(self, T, dt, sample_every=1e-3):
        '''
        Solve the system. Only call after having populated the system.

        Parameters
        ----------
        T : float
            The length of time to solve for.

        dt : float
            The solver timestep. It's good practice to keep this under half the period of the fastest gyro-orbit.

        sample_every : float
            The sample timestep.

        Returns
        -------
        None
        '''

        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')

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


    def add_time(self, T):
        '''
        Given that a file has been created for / loaded into the system, continue the simulation for T seconds from where it left off.
        Automatically saves the results to the file upon completion.

        Parameters
        ----------
        T : float
            The additional time for which to run the simulation.

        Returns
        -------
        None
        '''

        f = h5py.File(f'{self.file}.hdf5', 'a')

        hist = f['history']

        rr = hist['position']
        vv = hist['velocity']
        bb = hist['magnetic_field']
        ee = hist['electric_field']
        mm = hist['mass']
        qq = hist['charge']
        tt = hist['time']

        if hasattr(self, 'initial_conditions'):
            del self.initial_conditions

        self.populated = False

        self.initial_conditions          = np.empty((int(hist.attrs['num_particles']), 4, 3))
        self.initial_conditions[:, 0, :] = rr[:, -1, :]
        self.initial_conditions[:, 1, :] = vv[:, -1, :]
        self.initial_conditions[:, 2, :] = bb[:, -1, :]
        self.initial_conditions[:, 3, :] = ee[:, -1, :]

        if hasattr(self, 'particle_properties'):
            del self.particle_properties

        self.particle_properties       = np.empty((int(hist.attrs['num_particles']), 2))
        self.particle_properties[:, 0] = mm[:]
        self.particle_properties[:, 1] = qq[:]

        self.n = hist.attrs['num_particles']

        self.populated = True

        self.solved = False

        self.solve(T + float(hist.attrs['solve_timestep']), float(hist.attrs['solve_timestep']), float(hist.attrs['sample_timestep']))

        rr.resize((hist.attrs['num_particles'], hist.attrs['steps'] + np.shape(self.history)[1] - 1, 3))
        vv.resize((hist.attrs['num_particles'], hist.attrs['steps'] + np.shape(self.history)[1] - 1, 3))
        bb.resize((hist.attrs['num_particles'], hist.attrs['steps'] + np.shape(self.history)[1] - 1, 3))
        ee.resize((hist.attrs['num_particles'], hist.attrs['steps'] + np.shape(self.history)[1] - 1, 3))
        tt.resize((hist.attrs['steps'] + np.shape(self.history)[1] - 1, ))

        rr[:, hist.attrs['steps']:, :] = self.history[:, 1:, 0, :]
        vv[:, hist.attrs['steps']:, :] = self.history[:, 1:, 1, :]
        bb[:, hist.attrs['steps']:, :] = self.history[:, 1:, 2, :]
        ee[:, hist.attrs['steps']:, :] = self.history[:, 1:, 3, :]

        hist.attrs['steps'] = hist.attrs['steps'] + np.shape(self.history)[1] - 1

        tt[:] = np.arange(0, hist.attrs['steps']) * hist.attrs['sample_timestep']
        
        f.close()


    def add_particles(self, trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e), by_eq_pa=False):
        '''
        Given that a file has been created for / loaded into the system, add additional trials to the simulation results.
        Automatically saves the results to the file upon completion.

        Parameters
        ----------
        trials : int
            The number of particles to add to the system.

        r_dist() / L : function / float
            A 3D distribution function / float for the particles' positions (in m). The former is obtained through the currying functions in distributions.py.
            The latter is expected if by_eq_pa=True.

        e_dist() : function
            A 1D distribution function for the particles' energies (in eV). This is obtained through the currying functions in distributions.py.

        pitch_ang_dist() : function
            A 1D distribution function for the particles' pitch angles (in rad). This is obtained through the currying functions in distributions.py.

        phase_ang_dist() : function A 1D distribution function for the particles' phase angles (in rad). This is obtained through the currying functions in distributions.py.

        m_dist() : function, optional
            A 1D distribution function for the particles' mass (in kg). This is obtained through the currying functions in distributions.py. Defaults to the electron mass.

        q_dist() : function, optional
            A 1D distribution function for the particles' charge (in C). This is obtained through the currying functions in distributions.py. Defaults to the electron charge.

        by_eq_pa : bool, optional
            Whether the new particles should be distributed by equatorial or instantaneous pitch angle. Defaults to false.

        Returns
        -------
        None
        '''

        if by_eq_pa:
            self.populate_by_eq_pa(trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e))
        else:
            self.populate(trials, r_dist, e_dist, pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e))

        f = h5py.File(f'{self.file}.hdf5', 'a')

        hist = f['history']

        rr = hist['position']
        vv = hist['velocity']
        bb = hist['magnetic_field']
        ee = hist['electric_field']
        mm = hist['mass']
        qq = hist['charge']

        self.solve(float(hist.attrs['sample_timestep']) * float(hist.attrs['steps']), float(hist.attrs['solve_timestep']), float(hist.attrs['sample_timestep']))

        rr.resize((hist.attrs['num_particles'] + np.shape(self.history)[0], hist.attrs['steps'], 3))
        vv.resize((hist.attrs['num_particles'] + np.shape(self.history)[0], hist.attrs['steps'], 3))
        bb.resize((hist.attrs['num_particles'] + np.shape(self.history)[0], hist.attrs['steps'], 3))
        ee.resize((hist.attrs['num_particles'] + np.shape(self.history)[0], hist.attrs['steps'], 3))
        mm.resize((hist.attrs['num_particles'] + np.shape(self.history)[0], ))
        qq.resize((hist.attrs['num_particles'] + np.shape(self.history)[0], ))

        rr[hist.attrs['num_particles']:, :, :] = self.history[:, :, 0, :]
        vv[hist.attrs['num_particles']:, :, :] = self.history[:, :, 1, :]
        bb[hist.attrs['num_particles']:, :, :] = self.history[:, :, 2, :]
        ee[hist.attrs['num_particles']:, :, :] = self.history[:, :, 3, :]
        mm[hist.attrs['num_particles']:] = self.particle_properties[:, 0]
        qq[hist.attrs['num_particles']:] = self.particle_properties[:, 1]

        hist.attrs['num_particles'] = hist.attrs['num_particles'] + np.shape(self.history)[0]

        f.close()
