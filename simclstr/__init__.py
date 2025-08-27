"""
Top-level for simclstr clustering package.

This package provides pattern-oriented behavior clustering for time-series data.
End users should use the main functions: read_time_series, perform_clustering, and related utilities.
"""

from .clusterer import (
    read_time_series,
    simulate_from_vensim,
    perform_clustering
)

from .plotting import (
    plot_clusters,
    interactive_plot_clusters
)

from .experiment_controller import (
    experiment_controller
)

__all__ = [
    "read_time_series",
    "simulate_from_vensim", 
    "perform_clustering",
    "plot_clusters",
    "interactive_plot_clusters",
    "experiment_controller"
]

__version__ = "0.1.0"