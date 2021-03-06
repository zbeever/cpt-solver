import time

import numpy as np
from scipy import constants as sp
from numba import njit, jit
from tqdm import tqdm

from ngeopack import geopack as gp
from ngeopack import t89 as ext_t89
from ngeopack import t96 as ext_t96
from ngeopack import t01 as ext_t01
from ngeopack import t04 as ext_t04
from ngeopack import t07 as ext_t07

from cptsolver.utils import Re, inv_Re

def sum_field(field_1, field_2):
    '''
    Sums two fields together.

    Parameters
    ----------
    field_1(r, t=0.) : function
        A field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    field_2(r, t=0.) : function
        A field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    Returns
    -------
    field(r, t=0.) : function
        A field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    @njit
    def field(r, t=0.):
        return field_1(r) + field_2(r)

    return field


def interpolated_field(b_field, xs, ys, zs, ds, file_to_load=None, file_to_save=None, check_error=True):
    '''
    Given a field, a bounding volume (in Earth radii), and a step size (in Earth radii), returns an interpolated field.
    This method makes use of a uniform three dimensional grid of samples, interpolating them with cubic Hermite splines.
    The discrepancy between the original and interpolated field is estimated at the end of this function.

    Parameters
    ----------
    b_field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    xs : float[2]
        The boundaries along the x axis (in Earth radii).

    ys : float[2]
        The boundaries along the y axis (in Earth radii).

    zs : float[2]
        The boundaries along the z axis (in Earth radii).

    ds : float
        The grid sample size (in Earth radii).

    Returns
    -------
    b_field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    '''

    # Because tricubic interpolation requires one point preceeding and two points proceeding the cube of
    # interest, we start sampling one block before the specified lower limit and stop two blocks after
    # the specified upper limit
    x_list = np.arange(xs[0] - ds, (np.ceil((xs[1] - xs[0]) / ds) + 3) * ds + xs[0], ds) * Re
    y_list = np.arange(ys[0] - ds, (np.ceil((ys[1] - ys[0]) / ds) + 3) * ds + ys[0], ds) * Re
    z_list = np.arange(zs[0] - ds, (np.ceil((zs[1] - zs[0]) / ds) + 3) * ds + zs[0], ds) * Re

    # The 'further' point
    r0 = np.array([x_list[0], y_list[0], z_list[0]])

    # The number of samples
    n_xs = x_list.size
    n_ys = y_list.size
    n_zs = z_list.size

    # The magnetic field at each sample point
    b_vals = np.empty((n_xs, n_ys, n_zs, 3))

    if file_to_load != None:
        b_vals = np.load(file_to_load)
    else:
        print('Sampling the magnetic field...')

        time.sleep(0.33)

        for i in tqdm(range(len(x_list))):
            for j in range(len(y_list)):
                for k in range(len(z_list)):
                    r_temp = np.array([x_list[i], y_list[j], z_list[k]])
                    if (r_temp == 0.0).all():
                        r_temp += np.array([1e-6, 1e-6, 1e-6])
                    b_vals[i, j, k, :] = b_field(r_temp)

        if file_to_save != None:
            np.save(file_to_save, b_vals)

    @njit
    def CINT(u, pnm1, pn, pn1, pn2):
        '''
        The Catmull-Rom cubic Hermite spline interpolation function.

        Parameters
        ----------
        u : float
            The parameter specifying the point at which to find the interpolated value. This value's range
            is between 0 and 1 and represents the location of the point between p_n and p_n+1

        pnm1 : float
            The sample point twice preceeding the point of interest, p_n-1

        pn : float
            The sample point preceeding the point of interest, p_n

        pn1 : float
            The sample point proceeding the point of interest, p_n+1

        pn2 : float
            The sample point twice proceeding the point of interest, p_n+2

        Returns
        -------
        val : float
            The interpolated value at the point parameterized by u, p_n, and p_n+1.
        '''

        return 0.5 * (((-pnm1 + 3 * pn - 3 * pn1 + pn2) * u + (2 * pnm1 - 5 * pn + 4 * pn1 - pn2)) * u + (-pnm1 + pn1)) * u + pn

    @njit
    def t(i, j, z, n_z, s):
        '''
        The t function from the tricubic interpolation algorithm on the Wikipedia page: https://en.wikipedia.org/wiki/Tricubic_interpolation

        Parameters
        ----------
        i : int
            The x block.

        j : int
            The y block.

        z : float
            The normalized z location between block n_z and n_z + 1.

        n_z : int
            The z block preceeding the point of interest.

        s : float[:, :, :, 3]
            The array of magnetic field samples.

        Returns
        -------
        val : float[3]
            The z-interpolated magnetic field vector.
        '''

        return CINT(z, s[i, j, n_z - 1], s[i, j, n_z], s[i, j, n_z + 1], s[i, j, n_z + 2])

    @njit
    def u(i, y, z, n_y, n_z, s):
        '''
        The u function from the tricubic interpolation algorithm on the Wikipedia page: https://en.wikipedia.org/wiki/Tricubic_interpolation

        Parameters
        ----------
        i : int
            The x block.

        y : float
            The normalized y location between block n_y and n_y + 1.

        z : float
            The normalized z location between block n_z and n_z + 1.

        n_y : int
            The y block preceeding the point of interest.

        n_z : int
            The z block preceeding the point of interest.

        s : float[:, :, :, 3]
            The array of magnetic field samples.

        Returns
        -------
        val : float[3]
            The y- and z-interpolated magnetic field vector.
        '''

        return CINT(y, t(i, n_y - 1, z, n_z, s), t(i, n_y, z, n_z, s), t(i, n_y + 1, z, n_z, s), t(i, n_y + 2, z, n_z, s))

    @njit
    def field(r, t=0.0):
        '''
        An edited version of the f function from the tricubic interpolation algorithm on the Wikipedia page: https://en.wikipedia.org/wiki/Tricubic_interpolation

        Parameters
        ----------
        r : float[3]
            The location at which the magnetic field is to be evaluated.

        t : float, optional
            Usually the time at which the field should be evaluated. Here, this is a dummy variable.
            For some reason, this function runs 10x slower if called without t, so make sure to always
            call field(r, 0.0).

        Returns
        -------
        val : float[3]
            The interpolated magnetic field vector.
        '''

        r_temp = ((r - r0) / Re) / ds

        # Find the x, y, and z blocks preceeding the point of interest
        n_x, n_y, n_z = r_temp.astype(np.int64)

        # Find the normalized x, y, and z distances between the preceeding and proceeding blocks
        x, y, z = r_temp - np.array([n_x, n_y, n_z])

        return CINT(x, u(n_x - 1, y, z, n_y, n_z, b_vals), u(n_x, y, z, n_y, n_z, b_vals), u(n_x + 1, y, z, n_y, n_z, b_vals), u(n_x + 2, y, z, n_y, n_z, b_vals))

    n_samples = 10000
    error_max = -np.inf
    error_avg = 0.0
    loc = np.zeros(3)

    if check_error:
        print('Estimating error...')

        time.sleep(0.33)

        for i in tqdm(range(n_samples)):
            x = np.random.uniform(xs[0], np.amin([xs[1], -1.0]))
            y = np.random.uniform(ys[0], ys[1])
            z = np.random.uniform(zs[0], zs[1])

            r = np.array([x, y, z]) * Re
            b_actual = b_field(r, 0.0)
            b_interp = field(r, 0.0)
            error = np.linalg.norm(b_interp - b_actual) / np.linalg.norm(b_actual) * 100

            error_avg += error

            if error > error_max:
                error_max = error
                loc = r

        error_avg /= n_samples

        time.sleep(0.33)

        print(f'Average error is {error_avg:.4f}%')
        print(f'Maximum error is {error_max:.4f}% at r = {loc / Re}')

    return field


def zero_field():
    '''
    Zero field. Returns the zero vector.

    Parameters
    ----------
    None

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    @njit
    def field(r, t=0.):
        return np.zeros(3)

    return field


def uniform_field(strength, axis):
    '''
    Uniform field. Returns the same value at every point in space.

    Parameters
    ----------
    strength : float
        The strength of the field (in T or V/m).

    axis : float[3]
        Direction along which the field lines point.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    norm_axis = axis / np.linalg.norm(axis)
    field_vec = strength * norm_axis

    @njit
    def field(r, t=0.):
        return field_vec

    return field


def harris_cs_model(b0x, sigma, L_cs):
    '''
    Harris current sheet model, with the current sheet in the x-y plane.

    Parameters
    ----------
    b0x : float
        The minimum value of the field (in T).

    sigma : float
        A parameter marking the perturbation strength of the current sheet. b0x * sigma is the radius of curvature of the field in the x-y plane.

    L_cs : float
        The current sheet thickness (in m).

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    @njit
    def field(r, t=0.):
        return np.array([b0x * np.tanh(r[2] / L_cs), 0., sigma * b0x])

    return field


def magnetic_dipole(current, signed_area):
    '''
    Magnetic dipole field formed from a current loop.

    Parameters
    ----------
    current : float
        The value of the current (in A).

    signed_area : float[3]
        Normal vector to the plane of the loop whose length is numerically equal to the loop's enclosed area (in m^2).

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    # Magnetic moment
    m = current * signed_area

    @njit
    def field(r, t=0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = sp.mu_0 / (4 * np.pi)

        return k * (3 * np.dot(m, r_unit) * r_unit - m) * r_mag**(-3)

    return field


def electric_dipole(charge, displacement):
    '''
    Electric dipole field formed from two opposite charges.

    Parameters
    ----------
    charge : float
        The magnitude of one of the charges (in C).

    displacement : float[3]
        The vector pointing from the negative charge to the positive one (in m).

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    # Dipole moment
    p = charge * displacement

    @njit
    def field(r, t=0.):
        r_mag = np.linalg.norm(r)
        r_unit = r / r_mag
        k = 1 / (4 * np.pi * sp.epsilon_0)

        return k * (3 * np.dot(p, r_unit) * r_unit - p) * r_mag**(-3)

    return field


def earth_dipole_axis_aligned():
    '''
    Dipole model of Earth's magnetic field with the dipole moment oriented along the z axis.

    Parameters
    ----------
    None

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    # Earth's magnetic moment with constants absorbed.
    M = -8e15

    @njit
    def field(r, t=0.):
        [x, y, z] = r
        R = np.sqrt(x**2 + y**2 + z**2)

        B_x = 3 * M * (x * z) / (R**5)
        B_y = 3 * M * (y * z) / (R**5)
        B_z = M * (3 * z**2 - R**2) / (R**5)

        return np.array([B_x, B_y, B_z])

    return field


def earth_dipole(t0=4.01172e7):
    '''
    Geopack dipole model of Earth's magnetic field.

    Parameters
    ----------
    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    gp.recalc(t0)

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.dip(x_gsm, y_gsm, z_gsm))

        return field_int * 1e-9

    return field


def igrf(t0=4.01172e7):
    '''
    The IGRF model of Earth's magnetic field.

    Parameters
    ----------
    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    ps, a, g, h, rec = gp.recalc(t0)

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm, a, g, h, rec))

        return field_int * 1e-9

    return field


def t89(Kp, t0=4.0118e7, sw_v=np.array([-400., 0., 0.])):
    '''
    Model of Earth's magnetic field consisting of a superposition of the Tsyganenko 1989 model (DOI: 10.1016/0032-0633(89)90066-4) and the IGRF model.

    Parameters
    ----------
    Kp : int
        A mapping to the Kp geomagnetic activity index. Acceptable values range from 1 to 7, mapping to values between 0 and 6+, inclusive.

    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    sw_v : float[3], optional
        The solar wind velocity vector in GSE coordinates. Defaults to v = [-400, 0, 0]

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    ps, a, g, h, rec = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm, a, g, h, rec))
        field_ext = np.asarray(ext_t89.t89(Kp, ps, x_gsm, y_gsm, z_gsm))

        return (field_int + field_ext) * 1e-9

    return field


def t96(par, t0=4.0118e7, sw_v=np.array([-400., 0., 0.])):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 1996 model (DOI: 10.1029/96JA02735) and the IGRF model.

    Parameters
    ----------
    par : float[10]
        A 10-element list containing the model parameters. These are

        par[0] (Pdyn) : float
            The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.

        par[1] (Dst) : float
            The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values
            are measured in nT. A value less than -50 nT indicates high geomagnetic activity.

        par[2] (ByIMF) : float
            The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.

        par[3] (BzIMF) : float
            The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.

        par[4:9] : float[6]
            Not used.

    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    ps, a, g, h, rec = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm, a, g, h, rec))
        field_ext = np.asarray(ext_t96.t96(par, ps, x_gsm, y_gsm, z_gsm))

        return (field_int + field_ext) * 1e-9

    return field


def t01(par, t0=4.0118e7, sw_v=np.array([-400., 0., 0.])):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2001 model (DOI: 10.1029/2001JA000220) and the IGRF model.

    Parameters
    ----------
    par : float[10]
        A 10-element list containing the model parameters. These are

        par[0] (Pdyn) : float
            The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.

        par[1] (Dst) : float
            The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value
            less than -50 nT indicates high geomagnetic activity.

        par[2] (ByIMF) : float
            The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.

        par[3] (BzIMF) : float
            The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.

        par[4] (G1) : float
            A parameter capturing the cross-tail current's dependence on the solar wind. Mathematically, it is defined as the average
            of V*h(B_perp)*sin^3(theta/2) where V is the solar wind speed, B_perp is the transverse IMF component, theta is the IMF's
            clock angle, and h is a function that behaves as B_perp^2 for common values of the IMF. A typical value is 6.

        par[5] (G2) : float
            A parameter capturing the earthward / tailward shift of the tail current, defined as the average of a*V*Bs, where V is the
            solar wind speed, Bs is the southward component of the IMF (|Bz| for Bz < 0, 0 otherwise) and a = 0.005. A typical value is 10.

        par[6:9] : float[4]
            Not used.

    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    ps, a, g, h, rec = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm, a, g, h, rec))
        field_ext = np.asarray(ext_t01.t01(par, ps, x_gsm, y_gsm, z_gsm))

        return (field_int + field_ext) * 1e-9

    return field


def t04(par, t0=4.01e7, sw_v=np.array([-400., 0., 0.])):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2004 model (DOI: 10.1029/2004JA010798) and the IGRF model.

    Parameters
    ----------
    par : float[10]
        A 10-element list containing the model parameters. These are

        par[0] (Pdyn) : float
            The solar wind dynamic pressure in nPa. Typically in the range of 1 to 6 nPa.

        par[1] (Dst) : float
            The disturbance storm-time index, a measure of magnetic activity connected to the ring current. Values are measured in nT. A value
            less than -50 nT indicates high geomagnetic activity.

        par[2] (ByIMF) : float
            The y component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.

        par[3] (BzIMF) : float
            The z component of the interplanetary magnetic field in nT. Total strength usually ranges from 1 to 37 nT.

        par[4] (Wt1) : float
            Driving parameter of the inner part of the tail field. Saturates at 0.71 +- 0.05. Peak estimate: 4 to 12.

        par[5] (Wt2) : float
            Driving parameter of the outer part of the tail field. Saturates at 0.39 +- 0.05. Peak estimate: 3 to 7.

        par[6] (Ws): float
            Driving parameter of the axially symmetric part of the ring current. Saturates at 3.3 +- 0.5. Peak estimate: 4 to 15.

        par[7] (Wp): float
            Driving parameter of the partial ring current field. Saturates at 75 +- 30. Peak estimate 10 to 50.

        par[8] (Wb1) : float
            Driving parameter of the principle mode of the Birkeland current. Saturates at 6.4 +- 1.0. Peak estimate 7 to 30.

        par[9] (Wb2) : float
            Driving parameter of the secondary mode of the Birkeland current. Saturates at 0.88 +- 0.06. Peak estimate 20 to 100.

    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    ps, a, g, h, rec = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm, a, g, h, rec))
        field_ext = np.asarray(ext_t04.t04(par, ps, x_gsm, y_gsm, z_gsm))

        return (field_int + field_ext) * 1e-9

    return field


def t07(par, ps, pdyn, t0=4.01e7, sw_v=np.array([-400., 0., 0.])):
    '''
    A model of Earth's magnetic field consisting of a superposition of the Tsyganenko 2007 model (DOI: 10.1029/2004JA010798) and the IGRF model.

    Parameters
    ----------
    par : float[10]
        A 10-element list containing the model parameters. These are

    t0 : float, optional
        The time (in seconds). Used for time-varying fields. Defaults to a time where the dipole tilt is approximately 0 degrees.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    ps_, a, g, h, rec = gp.recalc(t0, sw_v[0], sw_v[1], sw_v[2])

    @njit
    def field(r, t=0.):
        x_gsm = r[0] * inv_Re
        y_gsm = r[1] * inv_Re
        z_gsm = r[2] * inv_Re

        field_int = np.asarray(gp.igrf_gsm(x_gsm, y_gsm, z_gsm, a, g, h, rec))
        field_ext = np.asarray(ext_t07.t07(par, ps, pdyn, x_gsm, y_gsm, z_gsm))

        return (field_int + field_ext) * 1e-9

    return field

def xz_slice(field_func):
    '''
    Given a field, this takes the x-z plane at the origin and extends it along the y-axis, effectively removing one dimension from the system.

    Parameters
    ----------
    field_func : function
        The original field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    @njit
    def field(r, t=0.):
        r_fixed = np.array([r[0], 0., r[2]])
        return field_func(r_fixed, t)

    return field


def evolving_harris_cs_model(b0x, b0z, L_cs, lambd=40, gam=1):
    '''
    A Harris current sheet evolving in time. Has a strengthening factor in the x-component to force mirroring.

    Parameters
    ----------
    b0x : float
        The x-component of the magnetic field (in T) at the x-y plane.

    b0z : float
        The z-component of the magnetic field (in T) at the x-y plane.

    L_cs(t) : function
        The function describing the change in current sheet thickness (in m) over time.

    lambd : float, optional
        The damping factor of the strengthening field. Reverts to the classical Harris model if set to None. Defaults to 40 (which forces mirroring at z ~ +-20 Re).

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    @njit
    def field(r, t=0.):
        z = r[2]
        L = L_cs(t)

        if lambd == None:
            Bx = b0x * np.tanh(z / L)
        else:
            Bx = b0x * np.tanh(z / L) + np.exp(-lambd) * np.sinh(z * inv_Re)**(2 * gam - 1)
        By = 0
        Bz = b0z

        return np.array([Bx, By, Bz])

    return field


def evolving_harris_induced_E(b0x, L_cs, e0y=0., eps=1e-6):
    '''
    The induced electric field associated with an evolving Harris current sheet.

    Parameters
    ----------
    b0x : float
        The x-component of the magnetic field (in T) at the x-y plane.

    L_cs(t) : function
        The function describing the change in current sheet thickness (in m) over time.

    e0y : float
        The static, background electric field.

    eps : float, optional
        The small value used to calculate the derivative of L_cs(t). Defaults to 1e-6. Should be at least as small as the timestep of the simulation.

    Returns
    -------
    field(r, t=0.) : function
        The field function. Accepts a position (float[3]) and time (float). Returns the field vector (float[3]) at that point in spacetime.
    '''

    @njit
    def field(r, t=0.):
        z = r[2]
        L = L_cs(t)
        dLdt = (L_cs(t + 0.5 * eps) - L_cs(t - 0.5 * eps)) / eps

        Ex = 0
        Ey = b0x * dLdt * ((z / L) * np.tanh(z / L) - np.log(np.cosh(z / L)) - np.log(2)) + e0y
        Ez = 0

        return np.array([Ex, Ey, Ez])

    return field
