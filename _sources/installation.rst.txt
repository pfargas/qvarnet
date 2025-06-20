Installation
================

To install the `qvarnet` library, one has to clone the repository and install the package.

Installation Steps
-----------------
1. Go to the `GitHub repository <https://github.com/pfargas/qvarnet>`_
2. In the green "Code" button, copy the URL of the repository.
3. Open a terminal and navigate to the directory where you want to clone the repository.
4. Write ``git clone <repository_url>`` to clone the repository. Follow the steps the terminal provides.
5. Navigate into the cloned repository directory:

.. code-block:: bash

    cd qvarnet

6. Install the package:

.. code-block:: bash

    pip install -e .

7. Open a python interpreter or a Jupyter notebook and import the package:

.. code-block:: python
    
    import qvarnet

8. To update the package, just type in the terminal:

.. code-block:: bash

    git pull

The package is now installed in the editable mode, meaning that any changes made to the source code will be reflected immediately without needing to reinstall the package.