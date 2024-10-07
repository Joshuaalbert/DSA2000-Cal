#!/usr/bin/env python
import os
from typing import List

from setuptools import find_packages
from setuptools import setup


def get_all_subdirs(top: str, *dirs) -> List[str]:
    """
    Get all subdirectories of a directory (recursively).

    Args:
        top: the top directory to start from
        *dirs: the subdirectories to search for

    Returns:
        a list of all subdirectories
    """
    dir = os.path.join(*dirs)
    top = os.path.abspath(top)
    data_dir = []
    for dir, sub_dirs, files in os.walk(os.path.join(top, dir)):
        data_dir.append(dir[len(top) + 1:])
    data_dir = list(filter(lambda x: not os.path.basename(x).startswith('_'), data_dir))

    data_dir = list(map(lambda x: os.path.join(x, '*'), data_dir)) + list(
        map(lambda x: os.path.join(x, '.large_files'), data_dir))

    return data_dir


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
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.11',
      )
