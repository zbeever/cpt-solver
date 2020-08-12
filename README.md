# Charged particle trajectory solver
Solver designed to calculate and plot the trajectories and intrinsic quantities of a system of charged particles in the presence of electric and magnetic fields. This solver is not self-consistent, i.e. there is no space charge between the particles. We assume the particles under consideration are sufficiently sparse so as not to affect their cousins' trajectories.

## Dependencies
To use this package, you'll need

* Numpy
* Scipy
* Numba
* Matplotlib
* H5py

You'll also need the Numba modification of the Python version of the geopack library. You can find it at https://github.com/zbeever/ngeopack

## Installation
Using the terminal, type `pip install {path-to-directory-containing-setup.py}` This package can be removed using `pip uninstall ngeopack`
