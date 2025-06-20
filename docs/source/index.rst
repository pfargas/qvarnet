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

Basic usage
 ==================================

 - In *.src/qvarnet/models* one can find the ANNs defined. 
 - In *.src/qvarnet/hamiltonians* one can find the Hamiltonians defined, as well as the base class for the Hamiltonians.
 - In *.src/qvarnet/samplers* one can find the samplers defined.

 In the notebooks folder one can find examples of how to use the package.

The general description is:
1. Define a model with the architecture you want.
2. Define a pytorch optimizer
3. TODO: Define a hamiltonian object
   - Now you have to define the kinetic and potential term separately.
4. Define a sampler object/define your discretization
5. Start the training loop
   - Set optimizer gradients to zero
   - perform a forward pass of the model
   - calculate the energy
   - Add to the loss the energy and all the contributions you want to take into account
   - Perform a backward pass
   - Step the optimizer
6. Repeat until convergence

API reference  
================
.. automodule:: qvarnet/samplersv2.py
   :members:
