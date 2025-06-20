Samplers
========

This section provides an overview of the samplers available in the `qvarnet` library. Samplers are used to generate samples from distributions or datasets for various purposes.

Available Samplers
------------------

.. automodule:: qvarnet.samplersv2
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

Below are examples of how to use the samplers in `qvarnet`.

.. code-block:: python

    from qvarnet.samplersv2 import MetropolisHastingsSampler

    # Initialize the sampler
    sampler = ExampleSampler(parameters)

    # Generate samples
    samples = sampler.sample()

    # Process the samples
    for sample in samples:
         print(sample)

For more details, refer to the documentation of individual sampler classes and methods.