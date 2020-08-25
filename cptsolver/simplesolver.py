import os
from multiprocessing import cpu_count, Pool

from math import cos, sin, sqrt, asin, acos

import tqdm
import h5py
import numpy as np
import scipy.constants as sp
from numba import njit, prange

from cptsolver.integrators import relativistic_boris
from cptsolver.distributions import delta
from cptsolver.utils import Re, field_line, b_along_path, velocity_vec, format_bytes
from cptsolver.fields import evolving_harris_cs_model, evolving_harris_induced_E


@njit
def L_cs(t):
    return 0.1 * Re

@njit
def simplesolve_traj(i, steps, dt, initial_conditions, particle_properties, integrator, downsample, b0):
    hist_indiv = np.zeros(int(steps // downsample))

    state_0 = initial_conditions[i, :, :]
    state_1 = np.zeros((4, 3))

    for j in range(steps):
        boundary = 5. + np.random.rand()
        v = state_0[1, :]
        b = state_0[2, :]

        v_mag = sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        b_mag = sqrt(b[0]**2 + b[1]**2 + b[2]**2)
        
        cos_alpha = (v[0] * b[0] + v[1] * b[1] + v[2] * b[2]) / (v_mag * b_mag)
        sin_alpha = sqrt(1.0 - cos_alpha**2)

        if j % downsample == 0:
            ind = int(j // downsample)
            hist_indiv[ind] = asin(sqrt(b0 / b_mag) * sin_alpha)

        state_1[:, :] = integrator(state_0, particle_properties[i, :], dt, j)
        state_0[:, :] = state_1

        if (state_0[0, 2] > boundary * Re and state_0[1, 2] > 0) or (state_0[0, 2] < -boundary * Re and state_0[1, 2] < 0):
            v_par = v_mag * cos_alpha * b / b_mag
            v_perp = v - v_par

            state_0[1, :] = -v_par + v_perp

    return hist_indiv

class simplesolver:
    def __init__(self, b0x, b0z, e0y, L_cs):
        self.e_field = evolving_harris_induced_E(b0x, L_cs, e0y)
        self.b_field = evolving_harris_cs_model(b0x, b0z, L_cs, lambd=None)
        self.B_0 = b0z

        self.downsample = 1

        self.populated = False
        self.solved = False
        self.loaded = False

        return

    
    def save(self, filename):
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

        eqpa = hist.create_dataset('eq_pitch_ang', np.shape(self.eqpa), maxshape=(None, None), dtype='float', compression='gzip')

        # Log the filename for use in other functions
        self.file = filename

        # Set the solved system's attributes
        hist.attrs['solve_timestep']  = self.dt
        hist.attrs['sample_timestep'] = self.dt * self.downsample
        hist.attrs['num_particles']   = self.n
        hist.attrs['steps']           = np.shape(self.eqpa)[1]

        # Copy the solved system to the file
        eqpa[:] = self.eqpa[:]
            
        # Close the file and mark the solver as having loaded the solved system (to prevent an accidental re-solve)
        f.close()
        self.loaded = True

        # Get the saved file's size and let the user know the process has been completed
        file_size = os.path.getsize(f'{self.file}.hdf5')
        formatted_file_size = format_bytes(file_size)
        print(f'Saved file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')

        return

    
    def populate_by_eq_pa(self, trials, E_dist, eq_pitch_ang_dist, phase_ang_dist, m_dist=delta(sp.m_e), q_dist=delta(-sp.e), t=0., planar=False):
        self.n = trials

        self.initial_conditions  = np.zeros((self.n, 4, 3))
        self.particle_properties = np.zeros((self.n, 2))

        for i in tqdm.tqdm(range(self.n)):
            K = E_dist()
            m = m_dist()
            q = q_dist()

            eq_pa = eq_pitch_ang_dist()

            # Get the magnetic field at the particle's mirror point.
            B_m = self.B_0 / sin(eq_pa)**2

            # Set placeholder values for the particle's initial location and initial magnetic field
            z = 0.
            B_c = np.linalg.norm(self.b_field(np.array([0., 0., z])))

            # To keep track of sign changes when finding the correct mirror point
            sign_changes = 0
            past_below = np.sign(B_c - B_m) < 0

            # While the current magnetic field strength is far away from the mirror point
            # or greater than the mirror point, change the particle's initial location
            while abs(B_c - B_m) > 1e-11 or B_c > B_m:
                # Checks if the current location has a field strength below the mirror point
                below = np.sign(B_c - B_m) < 0

                # If the previous field strength was below B_m while the current is above (or vice versa),
                # track a sign change 
                if below != past_below:
                    sign_changes += 1
                    past_below = below

                # Take ever smaller steps towards a field strength barely below B_m
                if below:
                    z += 10**(-sign_changes) * Re
                else:
                    z -= 10**(-sign_changes) * Re

                # Calculate the new current field strength
                B_c = np.linalg.norm(self.b_field(np.array([0., 0., z])))

                # If the current location is beyond 5 earth radii, we are out of the current sheet and can instantiate
                # a particle at a non-mirror point
                if z > 5 * Re:
                    break


            # Get the sine of the (near) mirror point pitch angle.
            b_ratio = sqrt(B_c / self.B_0)
            new_sin = b_ratio * sin(eq_pa)

            # Get the spawn point of the particle.
            r = np.array([0., 0., z])

            pitch_angle = asin(new_sin)
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
        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')

        if self.loaded:
            raise NameError('Cannot solve an already solved system. Use add_particles or add_time.')

        return simplesolve_traj(i, self.steps, self.dt, self.initial_conditions, self.particle_properties, self.integrator, self.downsample, self.B_0)

    
    def solve(self, T, dt, integrator=relativistic_boris, sample_every=1e-3):
        if not self.populated:
            raise NameError('System not yet populated. Cannot solve without initial conditions.')

        self.integrator = integrator(self.e_field, self.b_field)
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

        self.eqpa = np.array(results)

        self.solved = True

        return
