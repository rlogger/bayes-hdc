VSA Models
==========

The ``bayes_hdc.vsa`` module provides Vector Symbolic Architecture model implementations.

All models share the same API: ``bind``, ``bundle``, ``inverse``, ``similarity``, ``random``.

Base Class
----------

.. autoclass:: bayes_hdc.vsa.VSAModel
   :members:
   :undoc-members:

Binary Spatter Codes
--------------------

.. autoclass:: bayes_hdc.vsa.BSC
   :members:
   :undoc-members:

Multiply-Add-Permute
--------------------

.. autoclass:: bayes_hdc.vsa.MAP
   :members:
   :undoc-members:

Holographic Reduced Representations
-----------------------------------

.. autoclass:: bayes_hdc.vsa.HRR
   :members:
   :undoc-members:

Fourier HRR
-----------

.. autoclass:: bayes_hdc.vsa.FHRR
   :members:
   :undoc-members:

Binary Sparse Block Codes
--------------------------

.. autoclass:: bayes_hdc.vsa.BSBC
   :members:
   :undoc-members:

Cyclic Group Representation
----------------------------

.. autoclass:: bayes_hdc.vsa.CGR
   :members:
   :undoc-members:

Modular Composite Representation
---------------------------------

.. autoclass:: bayes_hdc.vsa.MCR
   :members:
   :undoc-members:

Vector-Derived Transformation Binding
--------------------------------------

.. autoclass:: bayes_hdc.vsa.VTB
   :members:
   :undoc-members:

Factory Function
----------------

.. autofunction:: bayes_hdc.vsa.create_vsa_model
