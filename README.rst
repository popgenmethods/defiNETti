
================
defiNETti
================

defiNETti is a program for performing Bayesian inference on exchangeable 
(or permutation-invariant) data via deep learning. In particular, it is
well-suited for population genetic applications.

.. contents:: :depth: 2

Installation instructions
=========================
Prerequisites:

1. Scientific distribution of Python 2.7 or 3, e.g. [Anaconda](http://continuum.io/downloads), [Enthought Canopy](https://www.enthought.com/products/canopy/)
2. Alternatively, custom installation of pip, the SciPy stack

To install, in the top-level directory of defiNETti (where "setup.py" lives), type::

$ pip install .


Simulation
===========
The simulator is a function that returns a single datapoint tuple of ``(data, label)``.

1. ``data`` - A numpy array (e.g. genotype matrix, images, or point clouds) of 2 or 3 dimensions.
2. ``label`` - A numpy array of 1 dimension associated with the particular data array.

Simulator Example
-----------------

An example simulator for inferring gaussian parameters::

    import numpy as np

    def simulator_gaussian():
        mu = np.random.beta(5, 10)
        sigma = np.random.uniform(6,10)
        data = np.random.normal(mu, sigma, (100,1))
        label = np.array([mu, sigma]) 

        return (data,label)


Neural Network
==============
The neural network building blocks in this program supports two types of layers:

1. Convolutional Layers - The syntax is written as ``('conv', <#width>, <#output depth>)``. Note that the height of the image patches is assumed to be 1 to enforce exchangeability.
2. Fully-connected Layers - The syntax is written as ``('fc',  <#nodes>)``.
3. Matrix Multiply Layers - The last layer of the ``h_net`` neural network in the regression task. The syntax is written as ``('matmul',)``.
4. Softmax Layers - The last layer of the ``h_net`` neural network in the classification task. The syntax is written as ``('softmax',)``.

For more information regarding the differences see: (http://cs231n.github.io/convolutional-networks/). The multiple layers can be combined in the form of a list with the first element corresponding to the first layer and so on. For example, ``[("fc",1024),("fc",1024), ('softmax',)]``.



Training
=========
The train command trains an exchangeable neural network using simulation-on-the-fly. The exchangeable neural network learns a permutation-invariant function mapping :math:`f` from the data :math:`X = (x_1, x_2, \ldots, x_n)` to the label or the posterior over the label :math:`\theta`. In order to ensure permutation invariance, the function can be decomposed as:

.. math::

  f(X) = (h \circ g)(\Phi(x_1), \Phi(x_2), \ldots , \Phi(x_n))

- :math:`\Phi` - a function parameterized by a neural network that applies to each row of the input data.
- :math:`g` - a permutation-invariant function (e.g. max, sort, or moments).
- :math:`h` - a function parameterized by a neural network that applies to the exchangeable feature representation :math:`g(\Phi(x_1), \Phi(x_2), \ldots , \Phi(x_n))`.

For regression tasks, the output is an estimate of the label :math:`\hat{\theta} = f(X)`. For classification tasks, the output is a posterior over labels :math:`\mathcal{P}_{\theta} = f(X)`.

Arguments
---------
- ``input_shape`` - A tuple specifying the shape of the data output by ``<simulator>`` or :math:`X`. For example, if the data is a ``50 x 50`` image, then the shape should be listed as ``(50,50)``. Note that we always enforce permutation invariance in the 1st dimension and only support 2 or 3 dimensions.
- ``output_shape`` - A tuple specifying the shape of the label output by ``<simulator>`` or :math:`\theta`. For example, if the label is a 5-length continuous vector then the shape should be listed as ``(5,)``. If the label is a discrete variable, the size of the dimension is the number of classes. Note that currently only classification of a single label is implemented. Only 1 dimensional labels are currently supported.
- ``simulator`` - A function which returns tuples of ``(data, label)`` as mentioned above.
- ``phi_net`` - A neural network parameterizing :math:`\Phi` shown above. The input syntax is shown in the Neural Network section above.
- ``g`` -  An operation parameterizing the permutation-invariant function :math:`g` as shown above. The supported options include `('max',), ('sort',), ('top_k', <k>),` or `('moments', <m1>, <m2>, ...)`
- ``h_net`` - A neural network parameterizing :math:`h` as shown above. The input syntax is the same as for `phi_net`.
- ``network_function`` - A function of tensorflow operations specifying the neural net if you want to create your own network (if present ignores phi_net, g, and h_net).
- ``loss`` - The loss function to choose to train your neural network. Either "cross-ent" for cross-entropy loss or "l2" for l2-loss or a user-defined tensorflow function.
- ``accuracy`` - The metric for measuring accuracy to output. Either "classification" for 0-1 loss accuracy, None for using loss function as accuracy, or a user-defined tensorflow function.
- ``num_batches`` - The number of iterations (or batches) of training to perform when training the neural network.
- ``batch_size`` -  The size of each batch.
- ``queue_capacity`` - The number of training examples to hold in the queue at once.
- ``verbosity`` - Print every accuracy every ``<verbosity>`` iterations.
- ``training_threads`` - The number of threads dedicated to training the network. 
- ``sim_threads`` - The number of threads dedicated to simulating data.
- ``save_path`` - The base filename to save the neural network. If None, the weights are not saved.

Note: How to include distances for the 3-dimension use case. Vector can simply be padded with a 1 in the second dimension.
Note: How to feed in simulators in python?
Note: Return accuracy values for training curves?

Testing
========
The test command takes in data and a trained neural network to output predictions.

Arguments
---------
- ``data`` - A list of numpy arrays on which to run the neural network. The dimension of each numpy array should be the same as the input_shape in ``train()``.
- ``model_path`` - Path to the basename where the network is stored, should be same as save_path in ``train()``.
- ``threads`` - Number of threads used for the tensorflow operations

Output
------
- ``output`` - A numpy array containing the network output for each input. The dimensions of the numpy array are ``(<length of data list>, <output_shape[0]>)``.
