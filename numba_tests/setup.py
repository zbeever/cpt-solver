from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["integrators.pyx", "fields.pyx", "system.pyx"], annotate=True)
)
