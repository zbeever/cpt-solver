import numpy as np
import scipy.constants as sp
from utils import *
from numba import njit
from constants import *

@njit
def J_to_eV(E):
    return 1.0 / sp.e * E


@njit
def eV_to_J(E):
    return sp.e * E


@njit
def dot(v, w):
    return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]


@njit
def gamma(v):
    return 1.0 / np.sqrt(1 - dot(v, v) / sp.c**2)

@njit
def grad(field, r, eps=1e-6):
    x_offset = np.array([eps, 0., 0.])
    y_offset = np.array([0., eps, 0.])
    z_offset = np.array([0, 0., eps])

    return np.array([(np.linalg.norm(field(r + x_offset)) - np.linalg.norm(field(r - x_offset))) / (2 * eps),
                     (np.linalg.norm(field(r + y_offset)) - np.linalg.norm(field(r - y_offset))) / (2 * eps),
                     (np.linalg.norm(field(r + z_offset)) - np.linalg.norm(field(r - z_offset))) / (2 * eps)
                    ])           


@njit
def jacobian(field, r, eps=1e-6):
    x_offset = np.array([eps, 0., 0.])
    y_offset = np.array([0., eps, 0.])
    z_offset = np.array([0, 0., eps])

    jac = np.zeros((3,3))
    jac[:, 0] = (field(r + x_offset) - field(r - x_offset)) / (2 * eps)
    jac[:, 1] = (field(r + y_offset) - field(r - y_offset)) / (2 * eps)
    jac[:, 2] = (field(r + z_offset) - field(r - z_offset)) / (2 * eps)

    return jac


@njit
def flc(field, r, eps=1e-6):
    x_offset = np.array([eps, 0.0, 0.0])
    y_offset = np.array([0.0, eps, 0.0])
    z_offset = np.array([0.0, 0.0, eps])
    
    b = field(r)
    b /= np.linalg.norm(b)

    fx1 = field(r + x_offset)
    fx1 /= np.linalg.norm(fx1)
    
    fx0 = field(r - x_offset)
    fx0 /= np.linalg.norm(fx0)
    
    fy1 = field(r + y_offset)
    fy1 /= np.linalg.norm(fy1)
    
    fy0 = field(r - y_offset)
    fy0 /= np.linalg.norm(fy0)
    
    fz1 = field(r + z_offset)
    fz1 /= np.linalg.norm(fz1)
    
    fz0 = field(r - z_offset)
    fz0 /= np.linalg.norm(fz0)

    J = np.zeros((3, 3))
    J[:, 0] = (fx1 - fx0) / (2 * eps)
    J[:, 1] = (fy1 - fy0) / (2 * eps)
    J[:, 2] = (fz1 - fz0) / (2 * eps)
    
    return (1.0 / np.linalg.norm(np.dot(J, b)))


@njit
def field_line(field, r, tol):
    def rk45_step(field, r, h, tol, direction):
        a = np.sign(direction)
        
        k1 = a * field(r)
        k1 /= np.linalg.norm(k1)
        k1 *= h

        k2 = a * field(r + 0.25 * k1)
        k2 /= np.linalg.norm(k2)
        k2 *= h

        k3 = a * field(r + 0.09375 * k1 + 0.28125 * k2)
        k3 /= np.linalg.norm(k3)
        k3 *= h

        k4 = a * field(r + 0.87938097405553 * k1 - 3.2771961766045 * k2 + 3.3208921256259 * k3)
        k4 /= np.linalg.norm(k4)
        k4 *= h

        k5 = a * field(r + 2.0324074074074 * k1 - 8 * k2 + 7.1734892787524 * k3 - 0.20589668615984 * k4)
        k5 /= np.linalg.norm(k5)
        k5 *= h

        k6 = a * field(r - 0.2962962962963 * k1 + 2 * k2 - 1.3816764132554 * k3 + 0.45297270955166 * k4 - 0.275 * k5)
        k6 /= np.linalg.norm(k6)
        k6 *= h

        y_plus_1 = r + 0.11574074074074 * k1 + 0.54892787524366 * k3 + 0.53533138401559 * k4 - 0.2 * k5
        z_plus_1 = r + 0.11851851851852 * k1 + 0.51898635477583 * k3 + 0.50613149034202 * k4 - 0.18 * k5 + 0.036363636363636 * k6

        t_plus_1 = z_plus_1 - y_plus_1
        h = 0.9 * h * min(max(np.sqrt(tol / (2 * np.linalg.norm(t_plus_1))), 0.3), 2)
        
        return z_plus_1, h

    rrb = np.zeros((1, 3))
    rrb[0] = r
    
    h = 1e5
    while True:
        r, h = rk45_step(field, r, h, tol, -1)

        if np.linalg.norm(r) <= Re:
            break

        k = np.zeros((1, 3))
        k[0] = r
        rrb = np.append(rrb, k, axis=0)
        
    r = np.copy(rrb[0])
    
    rrf = np.zeros((1, 3))
    rrf[0] = r
    
    while True:
        r, h = rk45_step(field, r, h, tol, 1)

        if np.linalg.norm(r) <= Re:
            break

        k = np.zeros((1, 3))
        k[0] = r
        rrf = np.append(rrf, k, axis=0)
        
    rr = np.append(rrf[::-1], rrb, axis=0)
    
    return rr


@njit
def b_along_path(field, rr):
    steps = len(rr[:, 0])
    field_vec = np.zeros((steps, 3))
    field_mag = np.zeros(steps)
    field_rad_mag = np.zeros(steps)
    
    for i in range(steps):
        vec = field(rr[i])
        
        field_vec[i] = vec
        field_mag[i] = np.linalg.norm(vec)
        field_rad_mag[i] = np.sqrt(vec[0]**2 + vec[1]**2)
        
    return field_vec, field_mag, field_rad_mag


@njit
def field_reversal(field, rr):
    b_vec, b_mag, b_rad_mag = b_along_path(field, rr)
    return rr[b_rad_mag.argmin()]


@njit
def param_by_eq_pa(field, rr, eq_pa):
    # Get the magnetic field along the field line and the location of its minimu
    b_vec, b_mag, b_rad_mag = b_along_path(field, rr)
    min_ind = b_rad_mag.argmin()
    
    # Determine Bmax, the field value closest to the particle's mirror point
    b_max = b_mag[min_ind] / np.sin(eq_pa)**2
    
    # Walk northward along the field line until we find the closest value to Bmax
    i = min_ind
    while b_mag[i] < b_max:
        if i <= 0:
            break
        i -= 1

    if i == 0:
        if b_mag[i] > b_max:
            i += 1

        r = rr[i, :]
        pa = np.arcsin(np.sqrt(b_mag[i] / b_mag[min_ind]) * np.sin(eq_pa))

        return r, pa
    
    x = (rr[i + 1, 0] - rr[i, 0]) / (b_mag[i + 1] - b_mag[i]) * (b_max - b_mag[i]) + rr[i, 0]
    y = (rr[i + 1, 1] - rr[i, 1]) / (b_mag[i + 1] - b_mag[i]) * (b_max - b_mag[i]) + rr[i, 1]
    z = (rr[i + 1, 2] - rr[i, 2]) / (b_mag[i + 1] - b_mag[i]) * (b_max - b_mag[i]) + rr[i, 2]
    r = np.array([x, y, z])
    pa = np.arcsin(np.sqrt(np.linalg.norm(field(r)) / b_mag[min_ind]) * np.sin(eq_pa))

    return r, pa


@njit
def adiabaticity(field, rr, K, m = sp.m_e, q = -sp.e):
    hist_new = np.zeros((len(rr[:, 0])))

    gamma_v = 1 + eV_to_J(K) / (m * sp.c**2)
    v = (sp.c / gamma_v) * np.sqrt(gamma_v**2 - 1)

    for i in range(len(rr[:, 0])):
        b = field(rr[i, :])

        rho_0 = gamma_v * m * v / (abs(q) * np.linalg.norm(b))
        
        R_c = flc(field, rr[i, :])
        hist_new[i] = rho_0 / R_c

    return hist_new


def plot_field(field, axis, nodes, x_lims, y_lims, size = (10, 10), t = 0.0):
    x = np.linspace(x_lims[0], x_lims[1], nodes)
    y = np.linspace(y_lims[0], y_lims[1], nodes)

    U, V = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=size)

    if axis_num[axis] == 0:
        for i in range(nodes):
            for j in range(nodes):
                W, U[i][j], V[i][j] = field(np.array([1e-20, X[i][j], Y[i][j]]), t)
                ax.set_xlabel('$y$')
                ax.set_ylabel('$z$')
    elif axis_num[axis] == 1:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], W, V[i][j] = field(np.array([X[i][j], 1e-20, Y[i][j]]), t)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$z$')
    elif axis_num[axis] == 2:
        for i in range(nodes):
            for j in range(nodes):
                U[i][j], V[i][j], W  = field(np.array([X[i][j], Y[i][j], 1e-20]), t)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')

    color = 2 * np.log(np.hypot(U, V))
    ax.streamplot(X, Y, U, V, color=color, linewidth=1, cmap=plt.cm.jet, density=2, arrowstyle='wedge', arrowsize=1.)

    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    plt.show()
