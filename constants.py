axis_num = {'x' : 0,
            'y' : 1,
            'z' : 2}

mu0 = 1.256e-6 # H/m
epsilon0 = 8.854e-12 # F/m
me = 9.109e-31 # kg
mp = 1.672e-27 # kg
qe = 1.602e-19 # C
c = 3e8 # m/s
Re = 6.371e6 # m

def gamma(v):
    return (1 - (v/c)**2)**(-0.5)
