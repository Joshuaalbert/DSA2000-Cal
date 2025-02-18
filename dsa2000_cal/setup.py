#!/usr/bin/env python

from setuptools import setup


def load_requirements(file_name):
    with open(file_name, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "casa": load_requirements("requirements-casa.txt"),
        "geo": load_requirements("requirements-geo.txt"),
        "notebooks": load_requirements("requirements-notebooks.txt"),
        "tests": load_requirements("requirements-tests.txt"),
    }
)
