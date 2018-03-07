
defiNETti is a program for performing Bayesian inference on exchangeable 
(or permutation-invariant) data via deep learning. In particular, it is
well-suited for population genetic applications.

.. contents:: :depth: 2

Installation instructions
=========================
Prerequisites:
- Scientific distribution of Python 2.7 or 3, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
- Alternatively, custom installation of pip, the SciPy stack

To install, in the top-level directory of defiNETti (where "setup.py" lives), type::
$ pip install .


Simulation
===========
The simulator is a function that returns a single datapoint tuple of ``(data, label)``.

1. ``data`` - A numpy array (e.g. genotype matrix, images, or point clouds) of 2 or 3 dimensions.
2. ``label`` - A numpy array of 1 dimension associated with the particular data array.

Simulator Example
-----------------

Training
=========
The train command trains an exchangeable neural network using simulation-on-the-fly. The exchangeable neural network learns a permutation-invariant function mapping :math:`f` from the data :math:`X = (x_1, x_2, \ldots, x_n)` to the label or the posterior over the label :math:`\theta`. In order to ensure permutation invariance, the function can be decomposed as:
.. math::

   f(X) = (h \cdot g)(\Phi(x_1), \Phi(x_2), \ldots , \Phi(x_n))

- :math:`\Phi` - a function parameterized by a neural network that applies to each row of the input data.
- :math:`g` - a permutation-invariant function (e.g. max, sort, or moments).
- :math:`h` - a function parameterized by a neural network that applies to the exchangeable feature representation :math:`g(\Phi(x_1), \Phi(x_2), \ldots , \Phi(x_n))`.

For regression tasks, the output is an estimate of the label :math:`\hat{\theta} = f(X)`. For classification tasks, the output is a posterior over labels :math:`\mathcal{P}_{\theta} = f(X)`.

Arguments
^^^^^^^^^


Testing
========