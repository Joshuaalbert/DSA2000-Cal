#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

install_requires = [
    'pydantic<2',
    'numpy',
    'ujson',
    'astropy',
    'jax',
    'jaxlib',
    'tables',
    'ducc0',
    'tomographic_kernel',
    'jaxns'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dsa2000_cal',
      version='1.0.0',
      description='DSA2000 calibration and forward modelling code',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/joshuaalbert/dsa2000-cal",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      install_requires=install_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      include_package_data=True,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.11',
      )
