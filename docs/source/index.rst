.. qvarnet documentation master file, created by
   sphinx-quickstart on Wed May 21 23:26:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qvarnet's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Explanation of the details
===================================

The package is designed to have different parts for a whole VMC workflow.

 * Models: The ANN models (pytorch modules). It is really important that the forward pass of these models has to have dimensions (batch_size, input_layer). In the case of 1D, 1particle systems, the input layer is just 1 neuron, so the vector of positions has to have the shape (discretization, 1).

 * Hamiltonians: Definition of the target Hamiltonians

 * Samplers: Responsible for sampling the points at which the Models are evaluated. 
 That is in ML the generation of the training data.
