axis_num = {'x' : 0,
            'y' : 1,
            'z' : 2}

mu0 = 1.25663706212e-6 # H/m
epsilon0 = 8.8541878128e-12 # F/m
me = 9.10938356e-31 # kg
mp = 1.6726219e-27 # kg
qe = 1.60217662e-19 # C
c = 299792458. # m/s
Re = 6.371e6 # m

def gamma(v):
    return (1 - (v / c) ** 2) ** (-0.5)

def J_to_eV(E):
    return 6.24150913e18 * E

def eV_to_J(E):
    return 1.60217662e-19 * E
