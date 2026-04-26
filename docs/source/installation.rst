Installation
============

Clone the repository and install the package in editable mode.

Conda Environment
-----------------

A pre-configured conda environment file is provided. From the repository root:

.. code-block:: bash

    conda env create -f environment_config.yaml
    conda activate jax

Package Installation
--------------------

1. Visit the `GitHub repository <https://github.com/pfargas/qvarnet>`_ and copy
   the clone URL from the green **Code** button.

2. Clone the repository:

   .. code-block:: bash

       git clone <repository_url>
       cd qvarnet

3. Install in editable mode (changes to the source are reflected immediately):

   .. code-block:: bash

       pip install -e .

4. Verify the installation:

   .. code-block:: python

       import qvarnet
       print(qvarnet.__version__ if hasattr(qvarnet, "__version__") else "OK")

Updating
--------

Because the package is installed in editable mode, a ``git pull`` is all that
is needed to update to the latest version:

.. code-block:: bash

    git pull

Optional: documentation dependencies
-------------------------------------

To build this documentation locally, install the docs extras:

.. code-block:: bash

    pip install -e ".[docs]"

CLI Entry Point
---------------

After installation the ``qvarnet`` command is available in your environment:

.. code-block:: bash

    qvarnet --help
