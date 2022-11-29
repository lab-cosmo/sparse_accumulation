.. role:: bash(code)
   :language: bash

Sparse accumulation
===================

.. include:: docs/abstract.rst

++++++++++++
Installation
++++++++++++

:bash:`python3 -m pip install .`

++++++++++++
Tests
++++++++++++

gpu tests:
:bash:`python3 -m pytest test_cpp_jit_cuda.py -vrA`

cpu tests:
:bash:`python3 -m pytest test_cpp_contiguous.py`
    
+++++++++++++
Documentation
+++++++++++++

Documentation can be found `here <https://lab-cosmo.github.io/sparse_accumulation/index.html#>`_
