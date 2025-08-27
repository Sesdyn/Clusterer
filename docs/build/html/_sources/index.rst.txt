Simclstr documentation
===========================

Welcome to the documentation for ``simclstr``. This library provides utilities for time series clustering with support for multiple distance metrics, pattern-based analysis, and interactive visualizations.

**Core Dependencies:**

* ``numpy>=2.0.0, <2.3.0``
* ``scipy>=1.16.0``
* ``matplotlib>=3.7.0``
* ``pandas>=2.0.0``
* ``pysd>=3.0.0``
* ``numba>=0.57.0``

.. toctree::
   :maxdepth: 2
   :caption: Library Contents:

   installation
   Main Clustering Functions <api/clusterer>
   Plotting Functions <api/plotting>
   Experiment Controller <api/experiment_controller>

.. toctree::
   :maxdepth: 2
   :caption: Examples and Tutorials:

   PySD Integration Example <pysd/pysd_notebook>
   BasicA Instance Example <basicA/basicA_notebook>