Simclstr documentation
===========================

Welcome to the documentation for ``simclstr``. This library provides utilities for time series clustering with support for multiple distance metrics, pattern-based analysis, and interactive visualizations.

**Core Dependencies:**

* ``numpy>=2.1.0``
* ``scipy>=1.14.1``
* ``pandas>=2.2.3``
* ``numba>=0.62.0``
* ``openpyxl>=3.1.0``
* ``xlsxwriter>=3.1.0``
* ``dash>=2.14.0``

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