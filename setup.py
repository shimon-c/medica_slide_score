from os import environ, rename, path
from setuptools import setup

exec(open('slidecore/__init__.py').read())
setup(
    version=__version__,
    name='slidecore',
)
