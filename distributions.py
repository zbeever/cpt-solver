import numpy as np

class Dist:
    def __init__(self):
        return

    def sample(self):
        raise NotImplementedError()

class Delta(Dist):
    """Delta distribution. Returns the value its given.

    Args:
    val_: The value to return. Can be of any type.
    """

    def __init__(self, val_):
        self.val = val_

    def sample(self):
        return self.val

class Uniform(Dist):
    """Floating point uniform distribution. Can be any number of dimensions.

    Args:
    min_val_: The minimum value.
    max_val_: The maximum value.
    dims_: The number of dimensions spanned by the distribution. Defaults to 1.
    """

    def __init__(self, min_val_, max_val_, dims_ = 1):
        self.min_val = min_val_
        self.max_val = max_val_
        self.dims = dims_

    def sample(self):
        return np.random.default_rng().uniform(self.min_val, self.max_val, self.dims)

class Normal(Dist):
    """Floating point normal distribution. Can be any number of dimensions.

    Args:
    mean_: The mean.
    std_dev_: The standard deviations.
    dims_: The number of dimensions spanned by the distribution. Defaults to 1.
    """

    def __init__(self, mean_, std_dev_, dims_ = 1):
        self.mean = mean_
        self.std_dev = std_dev_
        self.dims = dims_

    def sample(self):
        return np.random.default_rng().normal(self.mean, self.std_dev, self.dims)

class UniformOOM(Dist):
    """Distribution that gives equal weight to a range of orders of magnitude.

    Args:
    min_oom_: The minimum order of magnitude.
    max_oom_: The maximum order of magnitude.
    dims_: The number of dimensions spanned by the distribution. Defaults to 1.
    """

    def __init__(self, min_oom_, max_oom_, dims_ = 1):
        self.min_oom = min_oom_
        self.max_oom = max_oom_
        self.dims = dims_

    def sample(self):
        return 10**np.random.default_rng().uniform(self.min_oom, self.max_oom, self.dims)

class UniformSphere(Dist):
    """Uniform distribution on S2.

    Args:
    radius_: The radius of the sphere.
    offset_: The center of the sphere. Defaults to the zero vector.
    """

    def __init__(self, radius_, offset_ = np.array([0., 0., 0.])):
        self.radius = radius_
        self.offset = offset_

    def sample(self):
        point = np.random.default_rng().normal(0., 1, 3)
        return self.radius * (point / np.linalg.norm(point)) + self.offset

class UniformCircle(Dist):
    """Uniform distribution on S1.

    Args:
    radius_: The radius of the circle.
    normal_: The normal to the plane of the circle. Defaults to the positive z axis unit vector.
    offset_: The center of the circle. Defaults to the zero vector.
    """

    def __init__(self, radius_, normal_ = np.array([0., 0., 1.]), offset_ = np.array([0., 0., 0.])):
        self.radius = radius_
        self.theta = np.arccos(normal_[2] / np.linalg.norm(normal_))
        self.phi = np.arctan2(normal_[1], normal_[0])
        self.offset = offset_

    def sample(self):
        point_2d = np.random.default_rng().normal(0., 1, 2)
        point = self.radius * np.array([point_2d[0], point_2d[1], 0.]) / np.linalg.norm(point_2d)

        new_x = np.cos(self.theta) * np.cos(self.phi) * point[0] - np.sin(self.phi) * point[1]
        new_y = np.cos(self.theta) * np.sin(self.phi) * point[0] + np.cos(self.phi) * point[1]
        new_z = -np.sin(self.theta) * point[0]

        return np.array([new_x, new_y, new_z]) + self.offset

class UniformDisc(Dist):
    """Uniform distribution on the two dimensional disc.

    Args:
    radius_: The radius of the disc.
    normal_: The normal to the plane of the disc. Defaults to the positive z axis unit vector.
    offset_: The center of the disc. Defaults to the zero vector.
    """

    def __init__(self, radius_, normal_ = np.array([0., 0., 1.]), offset_ = np.array([0., 0., 0.])):
        self.radius = radius_
        self.theta = np.arccos(normal_[2] / np.linalg.norm(normal_))
        self.phi = np.arctan2(normal_[1], normal_[0])
        self.offset = offset_

    def sample(self):
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        disc_r = np.sqrt(u)
        disc_phi = 2 * np.pi * v

        point = self.radius * disc_r * np.array([np.cos(disc_phi), np.sin(disc_phi), 0.])

        new_x = np.cos(self.theta) * np.cos(self.phi) * point[0] - np.sin(self.phi) * point[1]
        new_y = np.cos(self.theta) * np.sin(self.phi) * point[0] + np.cos(self.phi) * point[1]
        new_z = -np.sin(self.theta) * point[0]

        return np.array([new_x, new_y, new_z]) + self.offset

class UniformRing(Dist):
    """Uniform distribution on the two dimensional ring.

    Args:
    inner_radius_: The inner radius of the ring.
    outer_radius_: The outer radius of the ring.
    normal_: The normal to the plane of the ring. Defaults to the positive z axis unit vector.
    offset_: The center of the ring. Defaults to the zero vector.
    """

    def __init__(self, inner_radius_, outer_radius_, normal_ = np.array([0., 0., 1.]), offset_ = np.array([0., 0., 0.])):
        self.inner_radius = inner_radius_
        self.outer_radius = outer_radius_
        self.theta = np.arccos(normal_[2] / np.linalg.norm(normal_))
        self.phi = np.arctan2(normal_[1], normal_[0])
        self.offset = offset_

    def sample(self):
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        ring_r = np.sqrt(u * (self.outer_radius**2 - self.inner_radius**2) + self.inner_radius**2)
        ring_phi = 2 * np.pi * v

        point = ring_r * np.array([np.cos(ring_phi), np.sin(ring_phi), 0.])

        new_x = np.cos(self.theta) * np.cos(self.phi) * point[0] - np.sin(self.phi) * point[1]
        new_y = np.cos(self.theta) * np.sin(self.phi) * point[0] + np.cos(self.phi) * point[1]
        new_z = -np.sin(self.theta) * point[0]

        return np.array([new_x, new_y, new_z]) + self.offset

class UniformPartialRing(Dist):
    """Uniform distribution on the two dimensional partial ring.

    Args:
    inner_radius_: The inner radius of the partial ring.
    outer_radius_: The outer radius of the partial ring.
    inner_angle_: The smallest angle of the partial ring.
    outer_angle_: The largest angle of the partial ring.
    normal_: The normal to the plane of the partial ring. Defaults to the positive z axis unit vector.
    offset_: The center of the partial ring. Defaults to the zero vector.
    """

    def __init__(self, inner_radius_, outer_radius_, inner_angle_, outer_angle_, normal_ = np.array([0., 0., 1.]), offset_ = np.array([0., 0., 0.])):
        self.inner_radius = inner_radius_
        self.outer_radius = outer_radius_
        self.inner_angle = inner_angle_
        self.outer_angle = outer_angle_
        self.theta = np.arccos(normal_[2] / np.linalg.norm(normal_))
        self.phi = np.arctan2(normal_[1], normal_[0])
        self.offset = offset_

    def sample(self):
        u, v = np.random.default_rng().uniform(0., 1., 2) 
        ring_r = np.sqrt(u * (self.outer_radius**2 - self.inner_radius**2) + self.inner_radius**2)
        ring_phi = v * (self.outer_angle - self.inner_angle) + self.inner_angle

        point = ring_r * np.array([np.cos(ring_phi), np.sin(ring_phi), 0.])

        new_x = np.cos(self.theta) * np.cos(self.phi) * point[0] - np.sin(self.phi) * point[1]
        new_y = np.cos(self.theta) * np.sin(self.phi) * point[0] + np.cos(self.phi) * point[1]
        new_z = -np.sin(self.theta) * point[0]

        return np.array([new_x, new_y, new_z]) + self.offset
