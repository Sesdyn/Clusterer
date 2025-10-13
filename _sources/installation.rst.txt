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

Simclstr builds on the following third party libraries:

* Numpy >= 2.1.0
* Scipy >= 1.14.1
* Pandas >= 2.2.3
* Numba >= 0.62.0
* Dash >= 2.14.0
* Xlsxwriter >= 3.1.0
* Openpyxl >= 3.1.0

These modules should build automatically if you are installing via pip. If you are building from source, or if pip fails to load them, they can be loaded with the same pip syntax as above.

Optional Dependencies
---------------------

For using `PySD <https://pysd.readthedocs.io/en/master/index.html>`_ for simulation:

* PySD >= 3.9.0

For using plotting functions:

* Plotly >= 5.14.0
* Matplotlib >= 3.10.0