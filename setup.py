from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("analytical_fields.pyx", compiler_directives = {'language_level' : "3"}, annotate=True),
    zip_safe = False
)
