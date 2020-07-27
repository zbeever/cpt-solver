import os
import h5py
import tqdm
from utils import *
from diagnostics import *

class Analyzer:
    def __init__(self, filename):
        if not os.path.exists(f'{filename}.hdf5'):
            raise NameError('No such file.')

        self.file = filename

        file_size = os.path.getsize(f'{filename}.hdf5')
        formatted_file_size = format_bytes(file_size)

        print(f'Loaded file {self.file}.hdf5 containing {formatted_file_size[0]:.2f} {formatted_file_size[1]} of information.')


    def __open(self):
        '''
        Opens the associated file and checks to make sure the derived_quantities field exists. Refrain from calling this function directly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        self.f = h5py.File(f'{self.file}.hdf5', 'a')

        self.hist = self.f['history']

        self.rr = self.hist['position']
        self.vv = self.hist['velocity']
        self.bb = self.hist['magnetic_field']
        self.ee = self.hist['electric_field']
        self.mm = self.hist['mass']
        self.qq = self.hist['charge']
        self.tt = self.hist['time']

        if 'derived_quantities' not in self.f.keys():
            self.f.create_group('derived_quantities')

        self.dvqt = self.f['derived_quantities']

        return


    def __close(self):
        '''
        Closes the associated file and frees up memory. Refrain from calling this function directly.

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

        del self.dvqt

        return


    def __write(self, label, shape, func, force_recalc):
        '''
        Writes a new derived quantity to the file and returns the result.
        This process is lazy in the sense that it will return whatever is on file unless you force a recalculation.

        Parameters
        ----------
        label (string): The name of the dataset.
        shape (tuple): The shape of the dataset.
        func(): A zero-argument function returning the derived quantity.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        qt_v (NxNx...xN numpy array): The quantity calculated.
        '''

        qt_v = np.empty(shape)

        if label in self.dvqt.keys():
            if self.dvqt[label].shape == shape and not force_recalc:
                qt_v = self.dvqt[label][:]
            else:
                results = func()
                self.dvqt[label].resize(np.shape(results))
                self.dvqt[label][:] = results
                qt_v = self.dvqt[label][:]
        else:
            self.dvqt.create_dataset(label, shape, maxshape=tuple([None for k in range(len(shape))]), dtype='float', compression='gzip')
            self.dvqt[label][:] = func()
            qt_v = self.dvqt[label][:]

        return qt_v


    def __required(self, func_list, write=True, force_recalc=False):
        '''
        Ensures all listed quantities have been calculated.

        Parameters
        ----------
        func_list (list): A list of methods to run.
        force_recalc (bool): Whether each quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        None
        '''

        results = []

        for func in func_list:
            results.append(func(write=write, force_recalc=force_recalc))

        return tuple(results)


    def position(self):
        '''
        Returns the position vector (in m) at each point along each particle's trajectory.

        Parameters
        ----------
        None

        Returns
        -------
        rr_v (NxM numpy array): The position vector (in m) at each point along each particle's trajectory.
        '''

        self.__open()
        rr_v = self.rr[:]
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
        vv_v (NxM numpy array): The velocity vector (in m/s) at each point along each particle's trajectory.
        '''

        self.__open()
        vv_v = self.vv[:]
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
        bb_v (NxM numpy array): The magnetic field (in T) at each point along each particle's trajectory.
        '''

        self.__open()
        bb_v = self.bb[:]
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

        self.__open()
        ee_v = self.ee[:]
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

        self.__open()
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

        self.__open()
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

        self.__open()
        tt_v = self.tt[:]
        self.__close()

        return tt_v


    def r_mag(self, write=False, force_recalc=False):
        '''
        Returns the distance (in m) of each particle from the origin.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc). Defaults to false.

        Returns
        -------
        r_mag_v (NxM numpy array): The distance (in m) of each particle from the origin for each timestep.
        '''

        self.__open()

        def r_mag_curried():
            return position_mag(self.rr[:])

        if not write:
            return r_mag_curried()

        r_mag_v = self.__write('r_mag', self.rr.shape[0:2], r_mag_curried, force_recalc)
        self.__close()

        return r_mag_v


    def b_mag(self, write=True, force_recalc=False):
        '''
        Returns the magnetic field strength (in T) at each point along each particle's trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        b_mag_v (NxM numpy array): The magnetic field strength (in T) at each particle's location at each timestep.
        '''

        self.__open()

        def b_mag_curried():
            return b_mag(self.bb[:])

        if not write:
            return b_mag_curried()

        b_mag_v = self.__write('b_mag', self.bb.shape[0:2], b_mag_curried, force_recalc)
        self.__close()

        return b_mag_v


    def v_par(self, write=True, force_recalc=False):
        '''
        Returns the velocity (in m/s) parallel to the background magnetic field at each point along each particle's trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        v_par_v (NxM numpy array): The velocity (in m/s) parallel to the background magnetic field at each particle's location at each timestep.
        '''

        self.__open()

        def v_par_curried():
            return velocity_par(self.vv[:], self.bb[:])

        if not write:
            return v_par_curried()

        v_par_v = self.__write('v_par', self.vv.shape[0:2], v_par_curried, force_recalc)
        self.__close()

        return v_par_v


    def v_perp(self, write=True, force_recalc=False):
        '''
        Returns the velocity (in m/s) perpendicular to the background magnetic field at each point along each particle's trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        v_perp_v (NxM numpy array): The velocity (in m/s) perpendicular to the background magnetic field at each particle's location at each timestep.
        '''

        self.__open()

        def v_perp_curried():
            return velocity_perp(self.vv[:], self.bb[:])

        if not write:
            return v_perp_curried()

        v_perp_v = self.__write('v_perp', self.vv.shape[0:2], v_perp_curried, force_recalc)
        self.__close()

        return v_perp_v


    def ke(self, write=True, force_recalc=False):
        '''
        Returns the kinetic energy (in eV) of each particle at each point along its trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        v_perp_v (NxM numpy array): The kinetic energy (in eV) of each particle at each timestep.
        '''

        self.__open()

        def ke_curried():
            return kinetic_energy(self.vv[:], self.mm[:])  

        if not write:
            return ke_curried()

        ke_v = self.__write('ke', self.vv.shape[0:2], ke_curried, force_recalc)
        self.__close()

        return ke_v


    def moment(self, write=True, force_recalc=False):
        '''
        Returns the magnetic moment (in MeV/G) of each particle at each point along its trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        moment_v (NxM numpy array): The magnetic moment (in MeV/G) of each particle at each timestep.
        '''

        v_perp_v, b_mag_v = self.__required([self.v_perp, self.b_mag], write, force_recalc)

        self.__open()

        def moment_curried():
            return magnetic_moment(v_perp_v[:], b_mag_v[:], self.mm[:])

        if not write:
            return moment_curried()

        moment_v = self.__write('moment', self.vv.shape[0:2], moment_curried, force_recalc)
        self.__close()

        return moment_v


    def pitch_ang(self, write=True, force_recalc=False):
        '''
        Returns the pitch angle (in radians) of each particle at each point along its trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        pitch_ang_v (NxM numpy array): The pitch angle (in radians) of each particle at each timestep.
        '''

        self.__open()

        def pitch_ang_curried():
            return pitch_angle(self.vv[:], self.bb[:])

        if not write:
            return pitch_ang_curried()

        pitch_ang_v = self.__write('pitch_ang', self.vv.shape[0:2], pitch_ang_curried, force_recalc)
        self.__close()

        return pitch_ang_v

    
    def gyrorad(self, write=True, force_recalc=False):
        '''
        Returns the gyroradius (in m) of each particle at each point along its trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        gyrorad_v (NxM numpy array): The gyroradius (in m) of each particle at each timestep.
        '''

        ke_v, v_perp_v, b_mag_v = self.__required([self.ke, self.v_perp, self.b_mag], write, force_recalc)
        self.__open()

        def gyrorad_curried():
            return gyrorad(ke_v[:], v_perp_v[:], b_mag_v[:], self.mm[:], self.qq[:])

        if not write:
            return gyrorad_curried()

        gyrorad_v = self.__write('gyrorad', self.vv.shape[0:2], gyrorad_curried, force_recalc)
        self.__close()

        return gyrorad_v


    def gyrofreq(self, write=True, force_recalc=False):
        '''
        Returns the gyrofreq (in 1/s) of each particle at each point along its trajectory.

        Parameters
        ----------
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        gyrofreq_v (NxM numpy array): The gyrofrequency (in 1/s) of each particle at each timestep.
        '''

        ke_v, b_mag_v = self.__required([self.ke, self.b_mag], write, force_recalc)
        self.__open()

        def gyrofreq_curried():
            return gyrofreq(ke_v[:], b_mag_v[:], self.mm[:], self.qq[:])

        if not write:
            return gyrofreq_curried()

        gyrofreq_v = self.__write('gyrofreq', self.vv.shape[0:2], gyrofreq_curried, force_recalc)
        self.__close()

        return gyrofreq_v

    
    def eq_pitch_ang(self, b_field=None, unwrapped=False, write=True, force_recalc=False):
        '''
        Returns the equatorial pitch angle (in radians) of each particle at each point along its trajectory.

        Parameters
        ----------
        b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py). This is required for checking adiabaticity.
        unwrapped (bool): Whether the equatorial pitch angle should be displayed from 0 to pi / 2 or unwrapped and displayed from 0 to pi.
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        eq_pitch_ang_v (NxM numpy array): The equatorial pitch angle (in radians) of each particle at each timestep.
        '''

        pa_v, b_mag_v, gyrorad_v = self.__required([self.pitch_ang, self.b_mag, self.gyrorad], write=write, force_recalc=force_recalc)
        self.__open()

        if 'eq_pitch_ang' in self.dvqt.keys() and force_recalc == False:
            if self.dvqt['eq_pitch_ang'].shape == np.shape(pa_v[:]):
                return self.dvqt['eq_pitch_ang'][:]
        else:
            if b_field == None:
                raise NameError('Cannot calculate equatorial pitch angle without reference magnetic field with which to calculate adiabaticity.')

        def eq_pitch_ang_curried():
            return eq_pitch_angle(b_field, pa_v[:], b_mag_v[:], gyrorad_v[:], self.rr[:], unwrapped)

        if not write:
            return eq_pitch_ang_curried()

        eq_pitch_ang_v = self.__write('eq_pitch_ang', self.vv.shape[0:2], eq_pitch_ang_curried, force_recalc)
        self.__close()

        return eq_pitch_ang_v


    def gca(self, b_field=None, max_iterations=20, tolerance=1e-3, write=True, force_recalc=False):
        '''
        Returns the guiding center of each particle at each point along its trajectory.

        Parameters
        ----------
        b_field(r, t): The magnetic field function (this is obtained through the currying functions in fields.py).
        max_iterations (int): The maximum number of iterations to perform at each particle's timestep. Defaults to 20.
        tolerance (float): The preferred tolerance level. If this function does not reach the required tolerance level, the value with the lowest tolerance will be used. Defaults to 1e-3.
        write (bool): Whether the quantity should be written to the file. Defaults to false.
        force_recalc (bool): Whether the quantity should be recalculated (provided an array already exists on disc).

        Returns
        -------
        gca_v (NxMx3 numpy array): The guiding center of each particle at each timestep.
        '''

        self.__open()

        if 'gca' in self.dvqt.keys() and force_recalc == False:
            if self.dvqt['gca'].shape == self.rr.shape:
                return self.dvqt['gca'][:]
        else:
            if b_field == None:
                raise NameError('Cannot calculate guiding center trajectories without reference magnetic field.')

        def gca_curried():
            return gca(b_field, self.rr[:], self.vv[:], self.mm[:], self.qq[:], max_iterations, tolerance)

        if not write:
            return gca_curried()

        gca_v = self.__write('gca', self.rr.shape, gca_curried, force_recalc)
        self.__close()

        return gca_v
