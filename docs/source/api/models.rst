Models (``qvarnet.models``)
===========================

All wave function ansätze live here.  Every model inherits from
:class:`~qvarnet.models.base.BaseModel`, which itself extends ``flax.linen.Module``.
Models are registered under a string key in :data:`~qvarnet.models.registry.MODEL_REGISTRY`
using the :func:`~qvarnet.models.registry.register_model` decorator.

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Registry key
     - Class
   * - ``"mlp"``
     - :class:`~qvarnet.models.mlp.MLP`
   * - ``"exponential-deep-set"``
     - :class:`~qvarnet.models.deep_set.ExponentialDeepSet`
   * - ``"deep-set"``
     - :class:`~qvarnet.models.deep_set.DeepSet`

The exponential and fermionic models live in ``qvarnet.models.exponential``
and ``qvarnet.models.mlp_fermions`` respectively.

Registering a Custom Model
--------------------------

Create a Python file and decorate your class:

.. code-block:: python

    from qvarnet.models.registry import register_model
    from qvarnet.models.base import BaseModel
    from flax import linen as nn

    @register_model("my-custom-model")
    class MyModel(BaseModel):
        hidden: int = 64

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden)(x)
            return nn.Dense(1)(x)

        @classmethod
        def from_config(cls, model_args):
            return cls(hidden=model_args.get("hidden", 64))

        @classmethod
        def get_input_shape(cls, model_args, batch_size):
            return (batch_size, model_args["input_dim"])

Then pass it at runtime:

.. code-block:: bash

    qvarnet run --config my_config.json --custom-model my_model.py

Module reference
----------------

Base class
~~~~~~~~~~

.. automodule:: qvarnet.models.base
   :members:
   :undoc-members:
   :show-inheritance:

Registry
~~~~~~~~

.. automodule:: qvarnet.models.registry
   :members:
   :undoc-members:

MLP
~~~

.. automodule:: qvarnet.models.mlp
   :members:
   :undoc-members:
   :show-inheritance:

Deep Set
~~~~~~~~

.. automodule:: qvarnet.models.deep_set
   :members:
   :undoc-members:
   :show-inheritance:

Exponential Models
~~~~~~~~~~~~~~~~~~

.. automodule:: qvarnet.models.exponential
   :members:
   :undoc-members:
   :show-inheritance:

Fermionic MLP
~~~~~~~~~~~~~

.. automodule:: qvarnet.models.mlp_fermions
   :members:
   :undoc-members:
   :show-inheritance:

Custom Dense Layer
~~~~~~~~~~~~~~~~~~

.. automodule:: qvarnet.models.layers.custom_dense
   :members:
   :undoc-members:
   :show-inheritance:
