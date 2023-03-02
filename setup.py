from setuptools import setup, find_packages

setup(name='cwx',
      version='0.1',
      packages=find_packages(),
      description='Python package for ChesWx',
      license='GPL',
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Atmospheric Science',
                   'Topic :: Scientific/Engineering :: GIS',
                   'License :: OSI Approved :: GNU General Public License',
                   'Programming Language :: Python :: 3'],    
      install_requires=['matplotlib','netCDF4', 'numpy', 'pandas', 'Pillow', 'pycurl',
                        'seaborn','scipy', 'tzwhere', 'xarray'],
      python_requires='>=3')