import numpy as np
from numba import njit

def delta(val):
    """Delta distribution. Returns the value its given.

    Args:
    val: The value to return. Can be of any type.
    """

    def sample():
        return val

    return sample


def uniform(min_val, max_val, dims = 1):
    """Floating point uniform distribution. Can be any number of dimensions.

    Args:
    min_val: The minimum value.
    max_val: The maximum value.
    dims: The number of dimensions spanned by the distribution. Defaults to 1.
    """

    def sample():
        return np.random.default_rng().uniform(min_val, max_val, dims)

    return sample


def normal(mean, std_dev, dims = 1):
    """Floating point normal distribution. Can be any number of dimensions.

    Args:
    mean: The mean.
    std_dev: The standard deviations.
    dims: The number of dimensions spanned by the distribution. Defaults to 1.
    """

    def sample():
        return np.random.default_rng().normal(mean, std_dev, dims)

    return sample


def uniform_oom(min_oom, max_oom, dims = 1):
    """Distribution that gives equal weight to a range of orders of magnitude.

    Args:
    min_oom: The minimum order of magnitude.
    max_oom: The maximum order of magnitude.
    dims: The number of dimensions spanned by the distribution. Defaults to 1.
    """

    def sample():
        return 10 ** np.random.default_rng().uniform(min_oom, max_oom, dims)

    return sample

def uniform_sphere(radius, offset = np.zeros(3)):
    """Uniform distribution on S2.

    Args:
    radius: The radius of the sphere.
    offset: The center of the sphere. Defaults to the zero vector.
    """

    def sample():
        point = np.random.default_rng().normal(0., 1, 3)
        return radius * (point / np.linalg.norm(point)) + offset

    return sample

def uniform_circle(radius, normal = np.array([0., 0., 1.]), offset = np.zeros(3)):
    """Uniform distribution on S1.

    Args:
    radius: The radius of the circle.
    normal: The normal to the plane of the circle. Defaults to the positive z axis unit vector.
    offset: The center of the circle. Defaults to the zero vector.
    """

    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        point_2d = np.random.default_rng().normal(0., 1, 2)
        point = radius * np.array([point_2d[0], point_2d[1], 0.]) / np.linalg.norm(point_2d)

        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        return np.array([new_x, new_y, new_z]) + offset

    return sample


def uniform_disc(radius, normal = np.array([0., 0., 1.]), offset = np.zeros(3)):
    """Uniform distribution on the two dimensional disc.

    Args:
    radius: The radius of the disc.
    normal: The normal to the plane of the disc. Defaults to the positive z axis unit vector.
    offset: The center of the disc. Defaults to the zero vector.
    """

    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        disc_r = np.sqrt(u)
        disc_phi = 2 * np.pi * v

        point = radius * disc_r * np.array([np.cos(disc_phi), np.sin(disc_phi), 0.])

        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        return np.array([new_x, new_y, new_z]) + offset

    return sample


def uniform_ring(inner_radius, outer_radius, normal = np.array([0., 0., 1.]), offset = np.zeros(3)):
    """Uniform distribution on the two dimensional ring.

    Args:
    inner_radius: The inner radius of the ring.
    outer_radius: The outer radius of the ring.
    normal: The normal to the plane of the ring. Defaults to the positive z axis unit vector.
    offset: The center of the ring. Defaults to the zero vector.
    """

    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        ring_r = np.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        ring_phi = 2 * np.pi * v

        point = ring_r * np.array([np.cos(ring_phi), np.sin(ring_phi), 0.])

        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        return np.array([new_x, new_y, new_z]) + offset

    return sample


def uniform_partial_ring(inner_radius, outer_radius, inner_angle, outer_angle, normal = np.array([0., 0., 1.]), offset = np.zeros(3)):
    """Uniform distribution on the two dimensional partial ring.

    Args:
    inner_radius: The inner radius of the partial ring.
    outer_radius: The outer radius of the partial ring.
    inner_angle: The smallest angle of the partial ring.
    outer_angle: The largest angle of the partial ring.
    normal: The normal to the plane of the partial ring. Defaults to the positive z axis unit vector.
    offset: The center of the partial ring. Defaults to the zero vector.
    """

    theta = np.arccos(normal[2] / np.linalg.norm(normal))
    phi = np.arctan2(normal[1], normal[0])

    def sample():
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        ring_r = np.sqrt(u * (outer_radius**2 - inner_radius**2) + inner_radius**2)
        ring_phi = v * (outer_angle - inner_angle) + inner_angle

        point = ring_r * np.array([np.cos(ring_phi), np.sin(ring_phi), 0.])

        new_x = np.cos(theta) * np.cos(phi) * point[0] - np.sin(phi) * point[1]
        new_y = np.cos(theta) * np.sin(phi) * point[0] + np.cos(phi) * point[1]
        new_z = -np.sin(theta) * point[0]

        return np.array([new_x, new_y, new_z]) + offset

    return sample
