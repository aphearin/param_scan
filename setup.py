from setuptools import setup, find_packages


PACKAGENAME = "param_scan"
VERSION = "0.0.1"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin",
    author_email="ahearin@anl.gov",
    description="Package to run parameter scans of some model",
    long_description="Package to run parameter scans of some model",
    install_requires=["numpy", "scipy", "mpi4py", "h5py"],
    packages=find_packages(),
    url="https://github.com/aphearin/param_scan",
)
