axis_num = {'x' : 0,
            'y' : 1,
            'z' : 2}

Re = 6.371e6 # m
inv_Re = 1. / Re

def gamma(v):
    return (1 - (v / c) ** 2) ** (-0.5)

def J_to_eV(E):
    return 6.24150913e18 * E

def eV_to_J(E):
    return 1.60217662e-19 * E
