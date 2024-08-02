from os import environ, rename, path
from setuptools import setup

exec(open('slidems/__init__.py').read())
setup(
    version=__version__,
    name='slidems',
)
