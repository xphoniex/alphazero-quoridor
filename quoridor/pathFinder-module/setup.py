# $ python setup.py build
# $ sudo python setup.py install
# $ sudo rm -rf build
from distutils.core import setup, Extension
setup(name = 'pathFinder', version = '1.0',  \
   ext_modules = [Extension('pathFinder', ['pathFinder.cpp'], extra_compile_args=['-std=c++11'])])
