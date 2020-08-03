import os
import h5py
import numpy as np

from diagnostics import *

class analyzer:
    '''
    Used to analyze a solved system of particles.

    Methods
    -------
    __open(read_write='a')
        Opens the associated file and checks to make sure the derived_quantities field exists.

    __close()
        Closes the associated file and frees up memory.

    __required(func_list, recalc=False, *args)
        Ensures all functions listed in func_list have been run and returns their results.

    __prepare(label, shape, recalc)
        Grabs data from file or, failing that, creates a new database.

    position()
        Returns the position vector (in m) at each point along each particle's trajectory.

    velocity()
        Returns the velocity vector (in m/s) at each point along each particle's trajectory.

    magnetic_field()
        Returns the magnetic field (in T) at each point along each particle's trajectory.

    electric_field()

    mass()

    charge()

    time()

    r_mag(numba=False, recalc=False)

    b_mag(numba=False, recalc=False)

    v_par(numba=False, recalc=False)

    v_perp(numba=False, recalc=False)

    ke(numba=False, recalc=False)

    moment(numba=False, recalc=False)

    pitch_ang(numba=False, recalc=False)

    gyrorad(numba=False, recalc=False)

    gyrofreq(numba=False, recalc=False)

    eq_pitch_ang(b_field=None, unwrapped=False, recalc=False, recalc_all=False)

    gca(numba=False, recalc=False)

    moment_diff(delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False)

    eq_pitch_ang_diff(delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False)

    moment_trans(delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False)

    eq_pitch_ang_trans(delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False)
    '''

    def __init__(self, filename):
        '''
        Associated a file with the object. 

        Parameters
        ----------
        filename : string
            The path to the file. Do not include the extension.

        Returns
        -------
        None
        '''

        if not os.path.exists(f'{filename}.hdf5'):
            raise NameError('No such file.')

        self.file = filename

        file_size = os.path.getsize(f'{filename}.hdf5')
        formatted_file_size = format_bytes(file_size)

        print(f'Loaded file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')


    def __open(self, read_write='a'):
        '''
        Opens the associated file and checks to make sure the derived_quantities field exists.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        self.f = h5py.File(f'{self.file}.hdf5', read_write)

        self.hist = self.f['history']

        self.rr = self.hist['position']
        self.vv = self.hist['velocity']
        self.bb = self.hist['magnetic_field']
        self.ee = self.hist['electric_field']
        self.mm = self.hist['mass']
        self.qq = self.hist['charge']
        self.tt = self.hist['time']

        self.num_particles = self.hist.attrs['num_particles']
        self.steps = self.hist.attrs['steps']

        if 'derived_quantities' not in self.f.keys():
            self.f.create_group('derived_quantities')

        self.dvqt = self.f['derived_quantities']

        return


    def __close(self):
        '''
        Closes the associated file and frees up memory.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        self.f.close()

        del self.f

        del self.hist

        del self.rr
        del self.vv
        del self.bb
        del self.ee
        del self.mm
        del self.qq
        del self.tt

        del self.steps
        del self.num_particles

        del self.dvqt

        return


    def __required(self, func_list, recalc=False, *args):
        '''
        Ensures all functions listed in func_list have been run and returns their results.

        Parameters
        ----------
        func_list : function[N]
            A list of methods to run.

        recalc : bool, optional
            Whether the quantities requested should be recalculated (in the case they already exists on file). Defaults to false.

        *args : any
            If recalc=True, feeds additional arguments into method(s) being run.

        Returns
        -------
        results : (float[N, ..., M], ..., float[N, ..., M])
            Tuple containing the requested quantities in the same order as they were specified.
        '''

        if type(func_list) != list:
            return func_list(*args, recalc=recalc)

        results = []

        for func in func_list:
            results.append(func(*args, recalc=recalc))

        return tuple(results)


    def __prepare(self, label, shape, recalc):
        '''
        Grabs data from file or, failing that, creates a new database.

        Parameters
        ----------
        label : string
            The label of the quantity to use.

        shape : int(N)
            A tuple describing the shape of the dataset.

        recalc : bool
            Whether the requested quantity should be recalculated.

        Returns
        -------
        found_val : bool
            Whether the quantity could be found on file. 

        results : float[N, ..., M]
            An array containing the requested quantity.
        '''

        if label in self.dvqt.keys():
            if self.dvqt[label].shape == shape and not recalc:
                qt_v = self.dvqt[label][:]
                return True, qt_v
            else:
                del self.dvqt[label]

        self.dvqt.create_dataset(label, shape, maxshape=tuple([None for k in range(len(shape))]), dtype='float', compression='gzip')
        return False, None


    def position(self):
        '''
        Returns the position vector (in m) at each point along each particle's trajectory.

        Parameters
        ----------
        None

        Returns
        -------
        rr_v : float[N, M]
            The position vector (in m) at each point along each particle's trajectory.
        '''

        self.__open('r')
        rr_v = self.rr[:, :, :]

        self.__close()
        return rr_v


    def velocity(self):
        '''
        Returns the velocity vector (in m/s) at each point along each particle's trajectory.

        Parameters
        ----------
        None

        Returns
        -------
        vv_v : float[N, M]
            The velocity vector (in m/s) at each point along each particle's trajectory.
        '''

        self.__open('r')
        vv_v = self.vv[:, :, :]

        self.__close()
        return vv_v


    def magnetic_field(self):
        '''
        Returns the magnetic field (in T) at each point along each particle's trajectory.

        Parameters
        ----------
        None

        Returns
        -------
        bb_v : float[N, M]
            The magnetic field (in T) at each point along each particle's trajectory.
        '''

        self.__open('r')
        bb_v = self.bb[:, :, :]

        self.__close()
        return bb_v


    def electric_field(self):
        '''
        Returns the electric field (in V/m) at each point along each particle's trajectory.

        Parameters
        ----------
        None

        Returns
        -------
        ee_v (NxM numpy array): The electric field (in V/m) at each point along each particle's trajectory.
        '''

        self.__open('r')
        ee_v = self.ee[:, :, :]

        self.__close()
        return ee_v


    def mass(self):
        '''
        Returns the mass (in kg) of each particle.

        Parameters
        ----------
        None

        Returns
        -------
        mm_v (N numpy array): The mass (in kg) of each particle.
        '''

        self.__open('r')
        mm_v = self.mm[:]

        self.__close()
        return mm_v


    def charge(self):
        '''
        Returns the charge (in C) of each particle.

        Parameters
        ----------
        None

        Returns
        -------
        qq_v (N numpy array): The charge (in C) of each particle.
        '''

        self.__open('r')
        qq_v = self.qq[:]

        self.__close()
        return qq_v


    def time(self):
        '''
        Returns the time (in s) at each timestep.

        Parameters
        ----------
        None

        Returns
        -------
        tt_v (N numpy array): The time (in s) at each timestep.
        '''

        self.__open('r')
        tt_v = self.tt[:]

        self.__close()
        return tt_v


    def r_mag(self, numba=False, recalc=False):
        '''
        Returns the distance (in m) of each particle from the origin.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        r_mag_v (NxM numpy array): The distance (in m) of each particle from the origin for each timestep.
        '''

        self.__open()
        found_val, r_mag_v = self.__prepare('r_mag', self.rr.shape[0:2], recalc)

        if found_val:
            self.__close()
            return r_mag_v
        else:
            if numba:
                r_mag_v = mag(self.rr[:])
            else:
                r_mag_v = np.linalg.norm(self.rr[:], axis=2)

            self.dvqt['r_mag'][:] = r_mag_v
            self.__close()
            return r_mag_v


    def b_mag(self, numba=False, recalc=False):
        '''
        Returns the magnetic field strength (in T) at each point along each particle's trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        b_mag_v (NxM numpy array): The magnetic field strength (in T) at each particle's location at each timestep.
        '''

        self.__open()
        found_val, b_mag_v = self.__prepare('b_mag', self.bb.shape[0:2], recalc)

        if found_val:
            self.__close()
            return b_mag_v
        else:
            if numba:
                b_mag_v = mag(self.bb[:])
            else:
                b_mag_v = np.linalg.norm(self.bb[:], axis=2)

            self.dvqt['b_mag'][:] = b_mag_v
            self.__close()
            return b_mag_v


    def v_par(self, numba=False, recalc=False):
        '''
        Returns the velocity (in m/s) parallel to the background magnetic field at each point along each particle's trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        v_par_v (NxM numpy array): The velocity (in m/s) parallel to the background magnetic field at each particle's location at each timestep.
        '''

        self.__open()
        found_val, v_par_v = self.__prepare('v_par', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return v_par_v
        else:
            if numba:
                v_par_v = v_par(self.vv[:], self.bb[:])
            else:
                v_dot_b = np.sum(self.vv[:] * self.bb[:], axis=2)
                b_dot_b = np.sum(self.bb[:]**2, axis=2)
                v_par_v = np.abs(np.divide(v_dot_b, b_dot_b, out=np.zeros_like(v_dot_b), where=b_dot_b != 0) * np.linalg.norm(self.bb[:], axis=2))

            self.dvqt['v_par'][:] = v_par_v
            self.__close()
            return v_par_v


    def v_perp(self, numba=False, recalc=False):
        '''
        Returns the velocity (in m/s) perpendicular to the background magnetic field at each point along each particle's trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        v_perp_v (NxM numpy array): The velocity (in m/s) perpendicular to the background magnetic field at each particle's location at each timestep.
        '''

        self.__open()
        found_val, v_perp_v = self.__prepare('v_perp', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return v_perp_v
        else:
            if numba:
                v_perp_v = v_perp(self.vv[:], self.bb[:])
            else:
                v_dot_b = np.sum(self.vv[:] * self.bb[:], axis=2)
                b_dot_b = np.sum(self.bb[:]**2, axis=2)
                scale_factor = np.divide(v_dot_b, b_dot_b, out=np.zeros_like(v_dot_b), where=b_dot_b != 0)
                v_perp_v = np.linalg.norm(self.vv[:] - scale_factor[:, :, np.newaxis] * self.bb[:], axis=2)

            self.dvqt['v_perp'][:] = v_perp_v
            self.__close()
            return v_perp_v


    def ke(self, numba=False, recalc=False):
        '''
        Returns the kinetic energy (in eV) of each particle at each point along its trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        v_perp_v (NxM numpy array): The kinetic energy (in eV) of each particle at each timestep.
        '''

        self.__open()
        found_val, ke_v = self.__prepare('ke', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return ke_v
        else:
            if numba:
                ke_v = ke(self.vv[:], self.mm[:])
            else:
                gamma_v = 1.0 / np.sqrt(1.0 - np.sum(self.vv[:]**2, axis=2) / sp.c**2)
                ke_v = J_to_eV(np.array(self.mm[:])[:, np.newaxis] * sp.c**2 * (gamma_v[:] - 1.0))

            self.dvqt['ke'][:] = ke_v
            self.__close()
            return ke_v


    def moment(self, numba=False, recalc=False):
        '''
        Returns the magnetic moment (in MeV/G) of each particle at each point along its trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        moment_v (NxM numpy array): The magnetic moment (in MeV/G) of each particle at each timestep.
        '''

        self.__open()
        found_val, moment_v = self.__prepare('moment', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return moment_v
        else:
            if numba:
                moment_v = moment(self.vv[:], self.bb[:], self.mm[:])
            else:
                v_dot_b = np.sum(self.vv[:] * self.bb[:], axis=2)
                b_dot_b = np.sum(self.bb[:]**2, axis=2)
                scale_factor = np.divide(v_dot_b, b_dot_b, out=np.zeros_like(v_dot_b), where=b_dot_b != 0)
                v_perp_v = self.vv[:] - scale_factor[:, :, np.newaxis] * self.bb[:]
                v_perp_v_squared = np.sum(v_perp_v**2, axis=2)
                b_mag = np.sqrt(b_dot_b)
                moment_v = 0.5 * np.array(self.mm[:])[:, np.newaxis] * np.divide(v_perp_v_squared, b_mag, out=np.zeros_like(v_perp_v_squared), where=b_mag != 0) * 6.242e8

            self.dvqt['moment'][:] = moment_v
            self.__close()
            return moment_v


    def pitch_ang(self, numba=False, recalc=False):
        '''
        Returns the pitch angle (in radians) of each particle at each point along its trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        pitch_v (NxM numpy array): The pitch angle (in radians) of each particle at each timestep.
        '''

        self.__open()
        found_val, pitch_ang_v = self.__prepare('pitch_ang', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return pitch_ang_v
        else:
            if numba:
                pitch_ang_v = pitch_ang(self.vv[:], self.bb[:])
            else:
                v_dot_b = np.sum(self.vv[:] * self.bb[:], axis=2)
                v_mag = np.linalg.norm(self.vv[:], axis=2)
                b_mag = np.linalg.norm(self.bb[:], axis=2) 
                denom = v_mag * b_mag
                cos_pa = np.divide(v_dot_b, denom, out=np.zeros_like(v_dot_b), where=denom != 0)
                pitch_ang_v = np.arccos(cos_pa)

            self.dvqt['pitch_ang'][:] = pitch_ang_v
            self.__close()
            return pitch_ang_v

    
    def gyrorad(self, numba=False, recalc=False):
        '''
        Returns the gyroradius (in m) of each particle at each point along its trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        gyrorad_v (NxM numpy array): The gyroradius (in m) of each particle at each timestep.
        '''

        self.__open()
        found_val, gyrorad_v = self.__prepare('gyrorad', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return gyrorad_v
        else:
            if numba:
                gyrorad_v = gyrorad(self.vv[:], self.bb[:], self.mm[:], self.qq[:])
            else:
                v_dot_b = np.sum(self.vv[:] * self.bb[:], axis=2)
                b_dot_b = np.sum(self.bb[:]**2, axis=2)
                scale_factor = np.divide(v_dot_b, b_dot_b, out=np.zeros_like(v_dot_b), where=b_dot_b != 0)
                v_perp_v = np.linalg.norm(self.vv[:] - scale_factor[:, :, np.newaxis] * self.bb[:], axis=2)
                b_mag = np.sqrt(b_dot_b)
                gamma_v = 1.0 / np.sqrt(1.0 - np.sum(self.vv[:]**2, axis=2) / sp.c**2)
                gyrorad_v = (2 * np.pi * gamma_v[:] * np.array(self.mm[:])[:, np.newaxis]) / (2 * np.pi * np.abs(np.array(self.qq[:])[:, np.newaxis])) * np.divide(v_perp_v[:], b_mag, out=np.zeros_like(v_perp_v), where=b_mag != 0)

            self.dvqt['gyrorad'][:] = gyrorad_v
            self.__close()
            return gyrorad_v


    def gyrofreq(self, numba=False, recalc=False):
        '''
        Returns the gyrofreq (in 1/s) of each particle at each point along its trajectory.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        gyrofreq_v (NxM numpy array): The gyrofrequency (in 1/s) of each particle at each timestep.
        '''

        self.__open()
        found_val, gyrofreq_v = self.__prepare('gyrofreq', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return gyrofreq_v
        else:
            if numba:
                gyrofreq_v = gyrofreq(self.vv[:], self.bb[:], self.mm[:], self.qq[:])
            else:
                gamma_v = 1.0 / np.sqrt(1.0 - np.sum(self.vv[:]**2, axis=2) / sp.c**2)
                gyrofreq_v = np.abs(np.array(self.qq[:])[:, np.newaxis]) * np.linalg.norm(self.bb[:], axis=2) / (2 * np.pi * gamma_v[:] * np.array(self.mm[:])[:, np.newaxis])

            self.dvqt['gyrofreq'][:] = gyrofreq_v
            self.__close()
            return gyrofreq_v

    
    def eq_pitch_ang(self, b_field=None, unwrapped=False, recalc=False, recalc_all=False):
        '''
        Returns the equatorial pitch angle (in radians) of each particle at each point along its trajectory.

        Parameters
        ----------
        b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py). This is required for checking adiabaticity.
        unwrapped (bool): Whether the equatorial pitch angle should be displayed from 0 to pi / 2 or unwrapped and displayed from 0 to pi.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.
        recalc_all (bool): Whether all quantities which this quantity depends on should be recalculated. Defaults to false.

        Returns
        -------
        eq_pitch_ang_v (NxM numpy array): The equatorial pitch angle (in radians) of each particle at each timestep.
        '''

        recalc = recalc if recalc_all == False else True

        pa_v, b_mag_v, gyrorad_v = self.__required([self.pitch_ang, self.b_mag, self.gyrorad], recalc_all)
        self.__open()

        if b_field == None and ('eq_pitch_ang' not in self.dvqt.keys() or recalc == True):
            raise NameError('Cannot calculate equatorial pitch angle without reference magnetic field with which to calculate adiabaticity.')

        found_val, eq_pitch_ang_v = self.__prepare('eq_pitch_ang', self.vv.shape[0:2], recalc)

        if found_val:
            self.__close()
            return eq_pitch_ang_v
        else:
            by_min_b = True
            asym_eps = 4e-1
            adb_eps = 1e-2
            deg_eps = 3.5e-2

            # Find the well-defined mirror points
            mp_inds = np.diff(np.sign(pa_v - np.pi / 2), axis=1, append=0) != 0 # Get indices where the pitch angle crosses 90 degrees
            mp_inds[:, -1] = False                                            # Remove values accidentally set to true due to our use of np.diff

            # Get all contiguous slices between mirror points
            masked = np.ma.array(b_mag_v, mask=mp_inds)           # Mask B at mirror points
            slices = np.ma.notmasked_contiguous(masked, axis=1) # Find slices of B between mirror points

            # Mark crossing points as the points of weakest B in these contiguous slices
            x_inds = np.zeros(np.shape(mp_inds)).astype(bool)
            for i in range(self.num_particles):
                for sl in slices[i]:
                    if by_min_b:
                        x_inds[i, np.argmin(b_mag_v[i][sl]) + sl.start] = True # Mark the minimum magnetic field value in each contiguous slice flanking a mirror point
                    else:
                        x_inds[i, np.argmax(np.abs(pa_v[i][sl] - np.pi / 2)) + sl.start] = True # Mark the minimum magnetic field value in each contiguous slice flanking a mirror point

                
            # Adjacent crossing points should be spread symmetrically around a mirror point
            # If the first pair of crossings does not exhibit this symmetry, remove the first point for removal
            x_ind_0 = np.argmax(x_inds, axis=1)                     # Find the first crossing
            x_inds_alt = np.copy(x_inds)                            # So we don't alter our initial x_inds
            x_inds_alt[np.arange(self.num_particles), x_ind_0] = False # Set the first crossings to false
            x_ind_1 = np.argmax(x_inds_alt, axis=1)                 # Find the second crossing
            mp_ind_0 = np.argmax(mp_inds, axis=1)                   # Find the first mirror point
            numer = ((x_ind_1 - mp_ind_0) - (mp_ind_0 - x_ind_0)).astype(float)
            denom = ((x_ind_1 - mp_ind_0)).astype(float)
            asymmetry_beg = np.abs(np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0))

            # If the last pair of crossings does not exhibit this symmetry, mark the last point for removal
            x_ind_minus_1 = np.shape(x_inds)[1] - np.argmax(np.flip(x_inds, axis=1), axis=1) - 1
            x_inds_alt[np.arange(self.num_particles), x_ind_minus_1] = False
            x_ind_minus_2 = np.shape(x_inds)[1] - np.argmax(np.flip(x_inds_alt, axis=1), axis=1) - 1
            mp_ind_minus_1 = np.shape(mp_inds)[1] - np.argmax(np.flip(mp_inds, axis=1), axis=1) - 1
            numer = ((x_ind_minus_1 - mp_ind_minus_1) - (mp_ind_minus_1 - x_ind_minus_2)).astype(float)
            denom = ((mp_ind_minus_1 - x_ind_minus_2)).astype(float)
            asymmetry_end = np.abs(np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0))
                
            # Remove each first crossing that occurs before any mirror point and contributes to asymmetry
            to_change_beg = np.argwhere((asymmetry_beg > asym_eps) & (x_ind_0 < mp_ind_0))[:, 0]
            x_inds[to_change_beg, x_ind_0[to_change_beg]] = False

            # Remove each last crossing that occurs after all mirror points and contributes to asymmetry
            to_change_end = np.argwhere((asymmetry_end > asym_eps) & (x_ind_minus_1 > mp_ind_minus_1))[:, 0]
            x_inds[to_change_end, x_ind_minus_1[to_change_end]] = False

            # Get (possible) mirror points initially missed (pitch angles between 89 and 91 degrees)
            mp_within_1_deg = np.abs(pa_v - np.pi / 2) < deg_eps

            # Add missed mirror points at the beginning
            x_ind_0 = np.argmax(x_inds, axis=1)
            first_mp = np.argmax(mp_within_1_deg, axis=1)
            mp_before_x_ind_0 = np.argwhere((first_mp < x_ind_0) & (x_ind_0 < mp_ind_0))[:, 0]
            mp_inds[mp_before_x_ind_0, first_mp[mp_before_x_ind_0]] = True

            # Add missed mirror points at the end
            x_ind_minus_1 = np.shape(x_inds)[1] - np.argmax(np.flip(x_inds, axis=1), axis=1) - 1
            last_mp = np.shape(mp_inds)[1] - np.argmax(np.flip(mp_within_1_deg, axis=1), axis=1) - 1
            mp_after_x_ind_minus_1 = np.argwhere((last_mp > x_ind_minus_1) & (x_ind_minus_1 > mp_ind_minus_1))[:, 0]
            mp_inds[mp_after_x_ind_minus_1, last_mp[mp_after_x_ind_minus_1]] = True

            # Add points at the beginning whose regions show adiabatic behavior
            mp_ind_0 = np.argmax(mp_inds, axis=1) 
            check_beg = np.argwhere((mp_ind_0 > x_ind_0) & (x_ind_0 > 0) & (b_mag_v[:, 0] != 0))

            for i in check_beg:
                R_c = flc(b_field, self.rr[i, 0][0], self.tt[0])
                if gyrorad_v[i, 0] / R_c < adb_eps:
                    mp_inds[i, 0] = True

            # Add points at the end whose regions show adiabatic behavior
            mp_ind_minus_1 = np.shape(mp_inds)[1] - np.argmax(np.flip(mp_inds, axis=1), axis=1) - 1
            check_end = np.argwhere((mp_ind_minus_1 < x_ind_minus_1) & (x_ind_minus_1 < np.shape(x_inds)[1] - 1) & (b_mag_v[:, -1] != 0))
                    
            for i in check_end:
                R_c = flc(b_field, self.rr[i, -1][0], self.tt[-1])
                if gyrorad_v[i, -1] / R_c < adb_eps:
                    mp_inds[i, -1] = True
                    
            eq_pitch_ang_v = np.zeros((self.num_particles, self.steps))

            for i in range(np.shape(mp_inds)[0]):
                duplicate_mps = np.argwhere(mp_inds[i, :])[:, 0][np.argwhere(np.abs(np.diff(np.argwhere(mp_inds[i, :])[:, 0])) <= 1)[:, 0]]
                mp_inds[i, duplicate_mps] = False
                
                mp_ind = np.argwhere(mp_inds[i, :])[:, 0]
                x_ind = np.argwhere(x_inds[i, :])[:, 0]
                
                x_ind_max = len(x_ind)
                mp_ind_max = len(mp_ind)
                
                if x_ind_max == 0 or mp_ind_max == 0:
                    continue
                
                display_x = np.copy(x_ind)
                
                if x_ind[0] <= mp_ind[0]:
                    display_x[0] = 0
                else:
                    display_x = np.append(0, display_x)
                if x_ind[-1] > mp_ind[-1]:
                    display_x[-1] = self.steps - 1
                else:
                    display_x = np.append(display_x, self.steps - 1)
                    
                x_display_max = len(display_x)
                    
                b_mirror_avg = (b_mag_v[i, mp_ind] + b_mag_v[i, np.clip(mp_ind + 1, 0, self.steps - 1)]) * 0.5
                pa_avg = (pa_v[i, mp_ind] + pa_v[i, np.clip(mp_ind + 1, 0, self.steps - 1)]) * 0.5
                
                for j in range(mp_ind_max):
                    sin_eq_pa = np.sqrt(b_mag_v[i, x_ind[min(j, x_ind_max - 1)]] / (b_mirror_avg[j])) * np.sin(pa_avg[j])
                    eq_pa = np.arcsin(min(sin_eq_pa, 1))
                    eq_pitch_ang_v[i, display_x[j]:display_x[min(j + 1, len(display_x) - 1)] + 1] = eq_pa


            if unwrapped:
                eq_pitch_ang_v[pa_v > np.pi / 2] = np.pi - eq_pitch_ang_v[pa_v > np.pi / 2] 

            self.dvqt['eq_pitch_ang'][:] = eq_pitch_ang_v
            self.__close()
            return eq_pitch_ang_v


    def gca(self, numba=False, recalc=False):
        '''
        Returns the guiding center of each particle along a history.

        Parameters
        ----------
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.

        Returns
        -------
        gca_v (NxMx3 numpy array): The guiding center of each particle at each timestep.
        '''

        self.__open()
        found_val, gca_v = self.__prepare('gca', self.rr.shape, recalc)

        if found_val:
            self.__close()
            return gca_v
        else:
            if numba:
                gca_v = gca(self.bb[:], self.rr[:], self.vv[:], self.mm[:], self.qq[:])
            else:
                b_dot_b = np.sum(self.bb[:]**2, axis=2)
                b_cross_v = np.cross(self.bb[:], self.vv[:], axis=2)

                v_dot_v = np.sum(self.vv[:]**2, axis=2)
                gamma = 1.0 / np.sqrt(1.0 - v_dot_v / sp.c**2)
                gamma_over_b_dot_b = np.divide(gamma, b_dot_b, out=np.zeros_like(gamma), where=b_dot_b != 0)

                gca_v = self.rr[:] - gamma_over_b_dot_b[:, :, np.newaxis] * np.asarray(self.mm[:])[:, np.newaxis, np.newaxis] * b_cross_v[:] / np.asarray(self.qq[:])[:, np.newaxis, np.newaxis]

            self.dvqt['gca'][:] = gca_v
            self.__close()
            return gca_v


    def moment_diff(self, delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False):
        '''
        Returns the magnetic moment diffusion coefficient along a history.

        Parameters
        ----------
        delta_t (float): The timestep over which diffusion will be calculated.
        bins (int): The number of bins to use. Defaults to 100.
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.
        recalc_all (bool): Whether all quantities which this quantity depends on should be recalculated. Defaults to false.

        Returns
        -------
        bins_v (BINS numpy array): The bin labels.
        moment_diff_v (BINSxM numpy array): The magnetic moment diffusion coefficient at each timestep.
        '''

        recalc = recalc if recalc_all == False else True

        moment_v = self.__required(self.moment, recalc_all)
        self.__open()

        found_coef, moment_diff_v = self.__prepare('moment_diff_coef', (bins, self.steps), recalc)
        found_bins, bins_v = self.__prepare('moment_diff_bins', (bins, ), recalc)

        if delta_t == None and ('moment_diff_coef' not in self.dvqt.keys() or 'moment_diff_bins' not in self.dvqt.keys() or recalc == True):
            raise NameError('Cannot calculate diffusion without a timestep.')

        if found_coef and found_bins:
            self.__close()
            return bins_v, moment_diff_v
        else:
            if numba:
                bins_v, moment_diff_v = diffusion(moment_v, self.tt[:], delta_t, bins)
            else:
                dt = np.abs(self.tt[1] - self.tt[0])
                delta_t_ind = int(max(delta_t // dt, 1))
                
                end = np.roll(moment_v, -delta_t_ind, axis=1)
                end[:, -delta_t_ind:] = moment_v[:, -delta_t_ind:]

                ind_diff_coef = (end - moment_v)**2 / (2 * delta_t_ind * dt)
                
                max_val = np.amax(moment_v)
                bin_width = max_val / (bins - 1)
                binned_moment = (moment_v // bin_width).astype(int)
                
                unweighted_diff_coef = np.zeros((bins, self.steps))
                weights = np.zeros((bins, self.steps))

                for i in range(self.steps):
                    np.add.at(unweighted_diff_coef[:, i], binned_moment[:, i], ind_diff_coef[:, i])
                    np.add.at(weights[:, i], binned_moment[:, i], 1)
                    
                bins_v = np.linspace(0, max_val, bins)
                moment_diff_v = np.divide(unweighted_diff_coef, weights, out=np.zeros_like(unweighted_diff_coef), where=weights != 0)
            
            self.dvqt['moment_diff_bins'][:] = bins_v
            self.dvqt['moment_diff_coef'][:] = moment_diff_v
            self.__close()
            return bins_v, moment_diff_v


    def eq_pitch_ang_diff(self, delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False, b_field=None, unwrapped=False):
        '''
        Returns the equatorial pitch angle diffusion coefficient along a history.

        Parameters
        ----------
        delta_t (float): The timestep over which diffusion will be calculated.
        bins (int): The number of bins to use. Defaults to 100.
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.
        recalc_all (bool): Whether all quantities which this quantity depends on should be recalculated. Defaults to false.
        b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py). This is only used if recalc_all is set to True.
        unwrapped (bool): Whether the equatorial pitch angle should be displayed from 0 to pi / 2 or unwrapped and displayed from 0 to pi. This is only used if recalc_all is set to True.

        Returns
        -------
        bins_v (BINS numpy array): The bin labels.
        eq_pitch_ang_diff_v (BINSxM numpy array): The equatorial pitch angle diffusion coefficient at each timestep.
        '''

        recalc = recalc if recalc_all == False else True

        eq_pitch_ang_v = self.__required(self.eq_pitch_ang, recalc_all, b_field)

        self.__open()
        found_coef, eq_pitch_ang_diff_v = self.__prepare('eq_pitch_ang_diff_coef', (bins, self.steps), recalc)
        found_bins, bins_v = self.__prepare('eq_pitch_ang_diff_bins', (bins, ), recalc)

        if delta_t == None and ('eq_pitch_ang_diff_coef' not in self.dvqt.keys() or 'eq_pitch_ang_diff_bins' not in self.dvqt.keys() or recalc == True):
            raise NameError('Cannot calculate diffusion without a timestep.')

        if found_coef and found_bins:
            self.__close()
            return bins_v, eq_pitch_ang_diff_v
        else:
            if numba:
                bins_v, eq_pitch_ang_diff_v = diffusion(eq_pitch_ang_v, self.tt[:], delta_t, bins)
            else:
                dt = np.abs(self.tt[1] - self.tt[0])
                delta_t_ind = int(max(delta_t // dt, 1))
                
                end = np.roll(eq_pitch_ang_v, -delta_t_ind, axis=1)
                end[:, -delta_t_ind:] = eq_pitch_ang_v[:, -delta_t_ind:]

                ind_diff_coef = (end - eq_pitch_ang_v)**2 / (2 * delta_t_ind * dt)
                
                max_val = np.amax(eq_pitch_ang_v)
                bin_width = max_val / (bins - 1)
                binned_eq_pitch_ang = (eq_pitch_ang_v // bin_width).astype(int)
                
                unweighted_diff_coef = np.zeros((bins, self.steps))
                weights = np.zeros((bins, self.steps))

                for i in range(self.steps):
                    np.add.at(unweighted_diff_coef[:, i], binned_eq_pitch_ang[:, i], ind_diff_coef[:, i])
                    np.add.at(weights[:, i], binned_eq_pitch_ang[:, i], 1)
                    
                bins_v = np.linspace(0, max_val, bins)
                eq_pitch_ang_diff_v = np.divide(unweighted_diff_coef, weights, out=np.zeros_like(unweighted_diff_coef), where=weights != 0)
            
            self.dvqt['eq_pitch_ang_diff_bins'][:] = bins_v
            self.dvqt['eq_pitch_ang_diff_coef'][:] = eq_pitch_ang_diff_v
            self.__close()
            return bins_v, eq_pitch_ang_diff_v


    def moment_trans(self, delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False):
        '''
        Returns the magnetic moment transport coefficient along a history.

        Parameters
        ----------
        delta_t (float): The timestep over which transport will be calculated.
        bins (int): The number of bins to use. Defaults to 100.
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.
        recalc_all (bool): Whether all quantities which this quantity depends on should be recalculated. Defaults to false.

        Returns
        -------
        bins_v (BINS numpy array): The bin labels.
        moment_trans_v (BINSxM numpy array): The magnetic moment transport coefficient at each timestep.
        '''

        recalc = recalc if recalc_all == False else True

        moment_v = self.__required(self.moment, recalc_all)

        self.__open()
        found_coef, moment_trans_v = self.__prepare('moment_trans_coef', (bins, self.steps), recalc)
        found_bins, bins_v = self.__prepare('moment_trans_bins', (bins, ), recalc)

        if delta_t == None and ('moment_trans_coef' not in self.dvqt.keys() or 'moment_trans_bins' not in self.dvqt.keys() or recalc == True):
            raise NameError('Cannot calculate transport without a timestep.')

        if found_coef and found_bins:
            self.__close()
            return bins_v, moment_trans_v
        else:
            if numba:
                bins_v, moment_trans_v = transport(moment_v, self.tt[:], delta_t, bins)
            else:
                dt = np.abs(self.tt[1] - self.tt[0])
                delta_t_ind = int(max(delta_t // dt, 1))
                
                end = np.roll(moment_v, -delta_t_ind, axis=1)
                end[:, -delta_t_ind:] = moment_v[:, -delta_t_ind:]

                ind_trans_coef = end - moment_v
                
                max_val = np.amax(moment_v)
                bin_width = max_val / (bins - 1)
                binned_moment = (moment_v // bin_width).astype(int)
                
                unweighted_trans_coef = np.zeros((bins, self.steps))
                weights = np.zeros((bins, self.steps))

                for i in range(self.steps):
                    np.add.at(unweighted_trans_coef[:, i], binned_moment[:, i], ind_trans_coef[:, i])
                    np.add.at(weights[:, i], binned_moment[:, i], 1)
                    
                bins_v = np.linspace(0, max_val, bins)
                moment_trans_v = np.divide(unweighted_trans_coef, weights, out=np.zeros_like(unweighted_trans_coef), where=weights != 0)
            
            self.dvqt['moment_trans_bins'][:] = bins_v
            self.dvqt['moment_trans_coef'][:] = moment_trans_v
            self.__close()
            return bins_v, moment_trans_v


    def eq_pitch_ang_trans(self, delta_t=None, bins=100, numba=False, recalc=False, recalc_all=False, b_field=None):
        '''
        Returns the equatorial pitch angle transport coefficient along a history.

        Parameters
        ----------
        delta_t (float): The timestep over which transport will be calculated.
        bins (int): The number of bins to use. Defaults to 100.
        numba (bool): Whether the Numba version of the function should be used (as opposed to the Numpy version). Defaults to false.
        recalc (bool): Whether the quantity should be recalculated (in the case it already exists on file). Defaults to false.
        recalc_all (bool): Whether all quantities which this quantity depends on should be recalculated. Defaults to false.
        b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py). This is only used if recalc_all is set to True.

        Returns
        -------
        bins_v (BINS numpy array): The bin labels.
        eq_pitch_ang_trans_v (BINSxM numpy array): The equatorial pitch angle transport coefficient at each timestep.
        '''

        recalc = recalc if recalc_all == False else True

        eq_pitch_ang_v = self.__required(self.moment, recalc_all, b_field)

        self.__open()
        found_coef, eq_pitch_ang_trans_v = self.__prepare('eq_pitch_ang_trans_coef', (bins, self.steps), recalc)
        found_bins, bins_v = self.__prepare('eq_pitch_ang_trans_bins', (bins, ), recalc)

        if delta_t == None and ('eq_pitch_ang_trans_coef' not in self.dvqt.keys() or 'eq_pitch_ang_trans_bins' not in self.dvqt.keys() or recalc == True):
            raise NameError('Cannot calculate transport without a timestep.')

        if found_coef and found_bins:
            self.__close()
            return bins_v, eq_pitch_ang_trans_v
        else:
            if numba:
                bins_v, eq_pitch_ang_trans_v = transport(eq_pitch_ang_v, self.tt[:], delta_t, bins)
            else:
                dt = np.abs(self.tt[1] - self.tt[0])
                delta_t_ind = int(max(delta_t // dt, 1))
                
                end = np.roll(eq_pitch_ang_v, -delta_t_ind, axis=1)
                end[:, -delta_t_ind:] = eq_pitch_ang_v[:, -delta_t_ind:]

                ind_trans_coef = end - eq_pitch_ang_v
                
                max_val = np.amax(eq_pitch_ang_v)
                bin_width = max_val / (bins - 1)
                binned_eq_pitch_ang = (eq_pitch_ang_v // bin_width).astype(int)
                
                unweighted_trans_coef = np.zeros((bins, self.steps))
                weights = np.zeros((bins, self.steps))

                for i in range(self.steps):
                    np.add.at(unweighted_trans_coef[:, i], binned_eq_pitch_ang[:, i], ind_trans_coef[:, i])
                    np.add.at(weights[:, i], binned_eq_pitch_ang[:, i], 1)
                    
                bins_v = np.linspace(0, max_val, bins)
                eq_pitch_ang_trans_v = np.divide(unweighted_trans_coef, weights, out=np.zeros_like(unweighted_trans_coef), where=weights != 0)
            
            self.dvqt['eq_pitch_ang_trans_bins'][:] = bins_v
            self.dvqt['eq_pitch_ang_trans_coef'][:] = eq_pitch_ang_trans_v
            self.__close()
            return bins_v, eq_pitch_ang_trans_v
