import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='cptsolver',
    version='0.9.0',
    author='Zach Beever',
    author_email='zbeever@bu.edu',
    description='Charged particle trajectory solver and analyzer.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url= 'https://github.com/zbeever/cpt-solver',
    requires= ['numpy', 'scipy', 'matplotlib', 'numba', 'h5py'],
    license= 'MIT',
    keywords= ['computational physics','space physics','Tsyganenko model','simulation'],
    packages= setuptools.find_packages(),
    package_data={'':['*.txt','*.md']},
    classifiers= [
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    python_requires='>=3.6',
)
