Installation
=============

Installing via pip
--------------------

To install the simclstr package from the Python package index use the pip command:

.. code-block:: bash

   pip install simclstr

Installing with conda
---------------------

To install simclstr with conda, using the conda-forge channel, use the following command:

.. code-block:: bash

   conda install -c conda-forge simclstr

Required Dependencies
---------------------

Simclstr requires **Python X.X** or above.

Simclstr builds on the core Python data analytics stack, and the following third party libraries:

* Numpy >= 2.0.0, < 2.3.0
* Scipy >= 1.16.0
* Pandas >= 1.24.0
* Numba >= 0.57.0
* Matplotlib >= 3.7.0

These modules should build automatically if you are installing via pip. If you are building from source, or if pip fails to load them, they can be loaded with the same pip syntax as above.

Optional Dependencies
---------------------

For using `PySD <https://pysd.readthedocs.io/en/master/index.html>`_ for simulation:

* PySD >= 3.0.0

For using interactive plotting functionality:

* Plotly >= 6.3.0
* Dash >= 2.14.0
* Dash Bootstrap Components >= 1.5.0
* Dash Extensions >= 1.0.0
* Dash Mantine Components >= 0.12.0
* Dash Iconify >= 0.1.2
* Xarray >= 2023.1.0, < 2024.0.0
* Pandas <= 2.0.0