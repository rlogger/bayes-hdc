Installation
============

Requirements
------------

* Python 3.9+
* JAX 0.4.20+
* jaxlib 0.4.20+

JAX-HDC is currently in **alpha** and is not yet published to PyPI.

Installing from Source
----------------------

.. code-block:: bash

   git clone https://github.com/rlogger/jax-hdc.git
   cd jax-hdc
   pip install -e .

For development (testing, linting, type checking):

.. code-block:: bash

   pip install -e ".[dev]"

For running examples (matplotlib, scikit-learn):

.. code-block:: bash

   pip install -e ".[examples]"

GPU/TPU
-------

JAX automatically uses GPU when available. For CUDA:

.. code-block:: bash

   pip install --upgrade "jax[cuda12]"

For TPU, install JAX with TPU support as per the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.
