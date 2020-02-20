from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    version='0.0.0',
    packages=['posenet_wrapper'],
    package_dir={'': '.'}
)

setup(**d)