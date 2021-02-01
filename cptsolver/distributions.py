import numpy as np
from scipy import integrate
from numba import njit
from scipy.interpolate import interp1d

def delta(val):
    '''
    Delta distribution. Returns the value its given.

    Parameters
    ----------
    val : any
        The value to return.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    def sample():
        return val

    return sample


def arbitrary(f, min_val, max_val, reconstruction_samples=10000):
    '''
    Allows sampling from arbitrary functions.

    Parameters
    ----------
    f : function
        The function describing the PDF. This does not need to be normalized.

    min_val : float
        The minimum value in the domain.

    max_val : float
        The maximum value in the domain.

    reconstruction_samples : int, optional
        The number of samples to use in constructing the inverse CDF of f.
    '''

    xs = np.linspace(min_val, max_val, reconstruction_samples, endpoint=True)
    dx = xs[1] - xs[0]

    fs = f(xs)
    fs /= np.sum(fs) * dx

    F = np.cumsum(fs) * dx
    F_inv = interp1d(F, xs, kind='linear', fill_value='extrapolate')

    def sample():
        return F_inv(np.random.default_rng().uniform(0, 1))

    return sample


def uniform(min_val, max_val, dims=1):
    '''
    Floating point uniform distribution. Can be any number of dimensions.

    Parameters
    ----------
    min_val : float
        The minimum value.

    max_val : float
        The maximum value.

    dims : int, optional
        The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    def sample():
        if dims == 1:
            return np.random.default_rng().uniform(min_val, max_val, dims)[0]
        else:
            return np.random.default_rng().uniform(min_val, max_val, dims)

    return sample


def normal(mean, std_dev, dims=1):
    '''
    Floating point normal distribution. Can be any number of dimensions.

    Parameters
    ----------
    mean : float
        The mean.

    std_dev : float
        The standard deviation.

    dims : int, optional
        The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    def sample():
        return np.random.default_rng().normal(mean, std_dev, dims)

    return sample


def uniform_oom(min_oom, max_oom, dims=1):
    '''
    Distribution that gives equal weight to a range of orders of magnitude.

    Parameters
    ----------
    min_oom : float
        The minimum order of magnitude.

    max_oom : float
        The maximum order of magnitude.

    dims : int, optional
        The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    def sample():
        return 10**np.random.default_rng().uniform(min_oom, max_oom, dims)

    return sample


def uniform_sphere(radius, offset=np.zeros(3)):
    '''
    Uniform distribution on S2.

    Parameters
    ----------
    radius : float
        The radius of the sphere.

    offset, optional : float[3]
        The center of the sphere. Defaults to the zero vector.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    def sample():
        # The 3D normal distribution is dependent only on r and so must be spherically symmetric.
        point = np.random.default_rng().normal(0., 1, 3)

        # Project the chosen point to a sphere of the given radius, then offset the result.
        return radius * (point / np.linalg.norm(point)) + offset

    return sample


def uniform_circle(radius, normal=np.array([0., 0., 1.]), offset=np.zeros(3)):
    '''
    Uniform distribution on S1.

    Parameters
    ----------
    radius : float
        The radius of the circle.

    normal : float[3], optional
        The normal to the plane of the circle. Defaults to the positive z axis unit vector.

    offset : float[3], optional
        The center of the circle. Defaults to the zero vector.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    # The zenith and azimuthal angles the normal vector makes with respect to the default Cartesian coordinate system.
    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        # The 2D normal distribution is dependent only on r and so must be circularly symmetric.
        point_2d = np.random.default_rng().normal(0., 1, 2)

        # Project the chosen point to a circle of the given radius.
        point = radius * np.array([point_2d[0], point_2d[1], 0.]) / np.linalg.norm(point_2d)

        # Rotate the circle to align with the given normal.
        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        # Offset the result.
        return np.array([new_x, new_y, new_z]) + offset

    return sample


def uniform_disc(radius, normal=np.array([0., 0., 1.]), offset=np.zeros(3)):
    '''
    Uniform distribution on the two dimensional disc.

    Parameters
    ----------
    radius : float
        The radius of the disc.

    normal : float[3], optional
        The normal to the plane of the circle. Defaults to the positive z axis unit vector.

    offset : float[3], optional
        The center of the circle. Defaults to the zero vector.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    # The zenith and azimuthal angles the normal vector makes with respect to the default Cartesian coordinate system.
    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        # Derived via the inverse CDF method. For more information, see notes in my raytracing project.
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        disc_r = np.sqrt(u)
        disc_phi = 2 * np.pi * v

        point = radius * disc_r * np.array([np.cos(disc_phi), np.sin(disc_phi), 0.])

        # Rotate the disc to align with the given normal.
        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        # Offset the result.
        return np.array([new_x, new_y, new_z]) + offset

    return sample


def uniform_ring(inner_radius, outer_radius, normal=np.array([0., 0., 1.]), offset=np.zeros(3)):
    '''
    Uniform distribution on the two dimensional ring.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the ring.

    outer_radius : float
        The outer radius of the ring.

    normal : float[3], optional
        The normal to the plane of the circle. Defaults to the positive z axis unit vector.

    offset : float[3], optional
        The center of the circle. Defaults to the zero vector.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    # The zenith and azimuthal angles the normal vector makes with respect to the default Cartesian coordinate system.
    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        # Derived via the inverse CDF method. For more information, see notes in my raytracing project.
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        ring_r = np.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        ring_phi = 2 * np.pi * v

        point = ring_r * np.array([np.cos(ring_phi), np.sin(ring_phi), 0.])

        # Rotate the ring to align with the given normal.
        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        # Offset the result.
        return np.array([new_x, new_y, new_z]) + offset

    return sample


def uniform_partial_ring(inner_radius, outer_radius, inner_angle, outer_angle, normal=np.array([0., 0., 1.]), offset=np.zeros(3)):
    '''
    Uniform distribution on the two dimensional partial ring.

    Parameters
    ----------
    inner_radius : float
        The inner radius of the partial ring.

    outer_radius : float
        The outer radius of the partial ring.

    inner_angle : float
        The smallest angle of the partial ring.

    outer_angle : float
        The largest angle of the partial ring.

    normal : float[3], optional
        The normal to the plane of the circle. Defaults to the positive z axis unit vector.

    offset : float[3], optional
        The center of the circle. Defaults to the zero vector.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    # The zenith and azimuthal angles the normal vector makes with respect to the default Cartesian coordinate system.
    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        # Derived via the inverse CDF method. For more information, see notes in my raytracing project.
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        ring_r = np.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        ring_phi = v * (outer_angle - inner_angle) + inner_angle

        point = ring_r * np.array([np.cos(ring_phi), np.sin(ring_phi), 0.])

        # Rotate the ring to align with the given normal.
        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        # Offset the result.
        return np.array([new_x, new_y, new_z]) + offset

    return sample


def weighted_cosine(a, b, k, dims=1): 
    '''
    Weighted cosine distribution proportional to cos^k. This mimics a butterfly distribution.

    Parameters
    ----------
    a : float
        The smallest value.

    b : float
        The greatest value.

    k : float
        The power to raise the cosine function with.

    dims : int, optional
        The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    # Proportional PDF.
    def unweighted_cosine_pdf(x, a, b, k):
        if x < a or x > b:
            return 0
        return np.cos(np.pi / (b - a) * (x - (b - a) / 2) - a / (b - a) * np.pi)**k

    # Calculate the area of the proportional PDF for normalization purposes.
    constant = integrate.quad(lambda x: unweighted_cosine_pdf(x, a, b, k), a, b)[0]

    # Integrate the normalized PDF to obtain the CDF.
    xs = np.linspace(a, b, 100)
    ys = [integrate.quad(lambda y: unweighted_cosine_pdf(y, a, b, k), a, x)[0] / constant for x in xs]

    def sample():
        # We switch xs and ys and interpolate to sample from the inverse CDF.
        if dims == 1:
            return np.interp(np.random.default_rng().uniform(0, 1, dims)[0], ys, xs)
        else:
            return np.interp(np.random.default_rng().uniform(0, 1, dims), ys, xs)

    return sample


def power_law(ll, ul, delta, dims=1):
    '''
    Power law distribution following p(x) = A*x^delta.

    Parameters
    ----------
    ll : float
        The smallest value.

    ul : float
        The greatest value.

    delta : float
        The power to use.

    dims : int, optional
        The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    -------
    sample() : function
        Function with no arguments that returns a sample from the distribution.
    '''

    A = (delta + 1) / (ul**(delta + 1) - ll**(delta + 1))

    def sample():
        if dims == 1:
            u = np.random.default_rng().uniform(0, 1, dims)[0]
        else:
            u = np.random.default_rng().uniform(0, 1, dims)
        return ((delta + 1) / A * u + ll**(delta + 1))**(1 / (delta + 1))

    return sample
