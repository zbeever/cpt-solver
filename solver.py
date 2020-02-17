import numpy as np
import sys
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

axis_num = {'x' : 0,
            'y' : 1,
            'z' : 2}

mu0 = 1 # 1.25663706e-6 H/m
epsilon0 = 1 # 8.85418782e-12 F/m
me = 1 # 9.10938356e-31 kg
mp = 1 # 1.6726219e-27 kg
qe = 1 # 1.60217662e-19 C

class System:
    def __init__(self, electric_field, magnetic_field):
        self.e_field = electric_field
        self.b_field = magnetic_field
        self.particles = []
        self.solutions = []
        self.solved = False

    def lorentzForce(self, x, t, q, m):
        # x takes the form [rx, ry, rz, vx, vy, vz] where r is the input position vector and v is the input velocity vector

        r = np.array([x[0], x[1], x[2]])
        v = np.array([x[3], x[4], x[5]])
        inv_m = 1 / m

        a = q * (self.e_field.at(r, t) + np.cross(v, self.b_field.at(r, t))) * inv_m

        return [v[0], v[1], v[2], a[0], a[1], a[2]]

    def addParticle(self, particle):
        self.particles.append(particle)

    def solve(self, t):
        for particle in self.particles:
            self.solutions.append(odeint(self.lorentzForce, particle.stateVars(), t, args=(particle.charge(), particle.mass())))
        self.solved = True

    def plot(self):
        if self.solved is False:
            sys.exit("error: run system.solve() before plotting")
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for solution in self.solutions:
                plt.plot(solution[:, 0], solution[:, 1], solution[:, 2])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()

class Particle:
    def __init__(self, r, v, q, m):
        self.r = r
        self.v = v
        self.q = q
        self.m = m

    def stateVars(self):
        return [self.r[0], self.r[1], self.r[2], self.v[0], self.v[1], self.v[2]]

    def charge(self):
        return self.q

    def mass(self):
        return self.m

class Field:
    def __init__(self):
        return

    def at(self, r, t):
        return

class UniformField(Field):
    # A uniform field. Simply specify the strength and the axis it
    # should be parallel to, 'x', 'y', or 'z'
    
    def __init__(self, strength, axis):
        self.field = np.array([0, 0, 0])
        self.field[axis_num[axis]] = strength

    def at(self, r, t):
        return self.field

class DipoleField(Field):
    # A dipole field generated from two nearby charges. Relative to
    # the (electric) dipole moment, strength corresponds to the magnitude
    # of qd while axis refers to the direction of the dipole's axis of symmetry

    def __init__(self, strength, axis):
        self.moment = np.array([0, 0, 0])
        self.moment[axis_num[axis]] = strength

    def at(self, r, t):
        r_mag = np.linalg.norm(r)
        return ( r * (3 * np.dot(r, self.moment)) / (r_mag**5) - self.moment / (r_mag**3) )

class OscillatingField(Field):
    # Either a temporally or spatially oscillating field. Its parameters are its
    # strength, the axis it lies along, its frequency (in space or time), and
    # whether the field is temporally varying (True) or spatially varying (False)

    def __init__(self, strength, axis, freq, time=False):
        self.axis = axis_num[axis]
        self.field = np.array([0, 0, 0])
        self.field[self.axis] = strength
        self.freq = freq
        self.time = time

    def at(self, r, t):
        if self.time is True:
            return self.field * np.sin(2*np.pi*self.freq*t)
        else:
            return self.field * np.sin(2*np.pi*self.freq*r[self.axis])

if __name__ == '__main__':
    '''
    # This code is for plotting the visualization of a dipole field.

    xs = np.arange(-5, 6, 1.3)
    ys = np.arange(-5, 6, 1.3)
    zs = np.arange(-5, 6, 1.3)

    xv, yv, zv = np.meshgrid(xs, ys, zs)

    u = xv * (3 * zv) / (xv**2 + yv**2 + zv**2)**(2.5)
    v = yv * (3 * zv) / (xv**2 + yv**2 + zv**2)**(2.5)
    w = zv * (3 * zv) / (xv**2 + yv**2 + zv**2)**(2.5) - 1 / (xv**2 + yv**2 + zv**2)**(1.5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.quiver(xv, yv, zv, u, v, w, normalize=True, cmap='hsv')
    plt.show()
    '''

    # Setup the electric and magnetic fields of the system. Superposition to be implemented in the future.
    e_field = OscillatingField(0.2, 'y', 3, False)
    b_field = DipoleField(1, 'z')

    # Create the system to be studied.
    system = System(e_field, b_field)

    # Add particles. The first array is the particle's position, the second is the particle's velocity.
    # The third argument is the particle's charge and the fourth is its mass.
    system.addParticle( Particle(np.array([0.2, 0.2, 0]), np.array([-0.1, 0.3, 0.5]), -qe, me) )
    system.addParticle( Particle(np.array([0, 0.3, 0.3]), np.array([0.05, 0.0, -0.2]), qe, mp) )
    system.addParticle( Particle(np.array([-0.1, -0.2, -0.25]), np.array([0.1, 0.15, 0.2]), qe, mp) )

    # The list of times to solve for
    t = np.linspace(0, 40, 4001)

    # Call these last. They solve the trajectories and plot the solutions
    system.solve(t)
    system.plot()

