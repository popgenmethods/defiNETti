#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name='defiNETti',
      version='1.0.0',
      description='A Likelihood-Free Inference Framework for Population Genetic Data using Exchangeable Neural Networks',
      author='Jeffrey Chan, Valerio Perrone, Jeffrey P. Spence, Paul A. Jenkins, Sara Mathieson, Yun S. Song',
      author_email='chanjed@berkeley.edu, v.Perrone@warwick.ac.uk, spence.jeffrey@berkeley.edu, p.Jenkins@warwick.ac.uk, smathie1@swarthmore.edu, yss@berkeley.edu',
      install_requires=['numpy>=1.14.2','tensorflow==1.6.0', 'msprime==0.4.0'],
      )
