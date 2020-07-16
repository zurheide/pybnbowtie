# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

import bowtie

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='bowtie',
    version=bowtie.__version__,
    description='A library to map bow-tie method to bayesian networks',
    long_description=readme,
    author='Frank T. Zurheide',
    author_email='frank.zurheide@gmail.com',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
