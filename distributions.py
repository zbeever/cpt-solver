import numpy as np
from scipy import integrate
from numba import njit

def delta(val):
    '''
    Delta distribution. Returns the value its given.

    Parameters
    ==========
    val (any): The value to return.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
    '''

    def sample():
        return val

    return sample


def uniform(min_val, max_val, dims=1):
    '''
    Floating point uniform distribution. Can be any number of dimensions.

    Parameters
    ==========
    min_val (float): The minimum value.
    max_val (float): The maximum value.
    dims (int): The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
    ==========
    mean (float): The mean.
    std_dev (float): The standard deviations.
    dims (int): The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
    '''

    def sample():
        return np.random.default_rng().normal(mean, std_dev, dims)

    return sample


def uniform_oom(min_oom, max_oom, dims=1):
    '''
    Distribution that gives equal weight to a range of orders of magnitude.

    Parameters
    ==========
    min_oom (float): The minimum order of magnitude.
    max_oom (float): The maximum order of magnitude.
    dims (int): The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
    '''

    def sample():
        return 10**np.random.default_rng().uniform(min_oom, max_oom, dims)

    return sample


def uniform_sphere(radius, offset=np.zeros(3)):
    '''
    Uniform distribution on S2.

    Parameters
    ==========
    radius (float): The radius of the sphere.
    offset (3x1 numpy array): The center of the sphere. Defaults to the zero vector.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
    ==========
    radius (float): The radius of the circle.
    normal (3x1 numpy array): The normal to the plane of the circle. Defaults to the positive z axis unit vector.
    offset (3x1 numpy array): The center of the circle. Defaults to the zero vector.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
    ==========
    radius (float): The radius of the disc.
    normal (3x1 numpy array): The normal to the plane of the disc. Defaults to the positive z axis unit vector.
    offset (3x1 numpy array): The center of the disc. Defaults to the zero vector.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
    ==========
    inner_radius (float): The inner radius of the ring.
    outer_radius (float): The outer radius of the ring.
    normal (3x1 numpy array): The normal to the plane of the ring. Defaults to the positive z axis unit vector.
    offset (3x1 numpy array): The center of the ring. Defaults to the zero vector.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
    ==========
    inner_radius (float): The inner radius of the partial ring.
    outer_radius (float): The outer radius of the partial ring.
    inner_angle (float): The smallest angle of the partial ring.
    outer_angle (float): The largest angle of the partial ring.
    normal (3x1 numpy array): The normal to the plane of the partial ring. Defaults to the positive z axis unit vector.
    offset (3x1 numpy array): The center of the partial ring. Defaults to the zero vector.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
    ==========
    a (float): The smallest value.
    b (float): The greatest value.
    k (float): The power to raise the cosine function with.
    dims (int): The number of dimensions spanned by the distribution. Defaults to 1.

    Returns
    =======
    sample(): Function with no argument that returns 1 sample when called.
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
