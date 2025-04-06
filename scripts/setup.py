from setuptools import setup, find_packages
import toml

pyproj = toml.load('pyproject.toml')['tool']['poetry']

setup(name=pyproj['name'],
      version=pyproj['version'],
      packages=find_packages(),
      )
