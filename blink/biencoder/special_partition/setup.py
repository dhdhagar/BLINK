from distutils.core import setup, Extension
import Cython
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("special_partition.pyx"),
    include_dirs=[numpy.get_include()]
)

"""
Build instructions: 
------------------
> cd special_partition
> python setup.py build_ext --inplace
"""