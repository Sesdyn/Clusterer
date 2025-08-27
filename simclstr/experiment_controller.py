"""
Experimental control module for clustering experiments.

This module provides utilities for running controlled clustering experiments,
comparing different clustering methods, and generating comprehensive reports.
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import warnings
import xlsxwriter
from .plotting import plot_clusters
from .clusterer import read_time_series, perform_clustering

if TYPE_CHECKING:
    from .clusterer import TimeSeries


def _normalize_data(list_of_ts_objects: List['TimeSeries']) -> List['TimeSeries']:
    """
    Compute the normalized version of the time-series data such that
    y_i = (x_i - min(x)) / (max(x) - min(x))
    
    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects
        
    Returns
    -------
    List['TimeSeries']
        List of TimeSeries objects
    """
    for each_ts in list_of_ts_objects:
        each_ts.data = (each_ts.data - np.min(each_ts.data)) / (np.max(each_ts.data) - np.min(each_ts.data))

    return list_of_ts_objects


def _standardize_data(list_of_ts_objects: List['TimeSeries']) -> List['TimeSeries']:
    """
    Compute the standardized version of the time-series data such that
    y_i = (x_i - mean(x)) / std(x)
    
    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects
        
    Returns
    -------
    List['TimeSeries']
        List of TimeSeries objects
    """
    for each_ts in list_of_ts_objects:
        each_ts.data = (each_ts.data - np.mean(each_ts.data)) / (1 if np.std(each_ts.data) == 0 else np.std(each_ts.data))

    return list_of_ts_objects


def _compare_clusterings(list_of_ts_objects: List['TimeSeries']) -> Tuple[float, float]:
    """
    Compare two clusterings using Rand index and Jaccard index.
    
    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects
 
    Returns
    -------
    Tuple[float, float]
        Two numbers: first being the Rand index, second being the Jaccard index
    """
    clusters1 = [each_ts.cluster_id for each_ts in list_of_ts_objects]
    clusters2 = [each_ts.previous_cluster_id for each_ts in list_of_ts_objects]

    if len(clusters1) != len(clusters2):
        raise ValueError("Number of members in these two clusterings are not equal")

    c1 = np.array(clusters1)
    c2 = np.array(clusters2)

    n = len(c1)
    c1_matrix = c1[:, np.newaxis] == c1[np.newaxis, :]
    c2_matrix = c2[:, np.newaxis] == c2[np.newaxis, :]
    
    # Extract upper triangular part to get unique pairs
    # k=1 parameter excludes the diagonal
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    c1_same_pairs = c1_matrix[upper_tri_mask]
    c2_same_pairs = c2_matrix[upper_tri_mask]
    
    count_same_in_both = np.sum(c1_same_pairs & c2_same_pairs)
    count_different_in_both = np.sum(~c1_same_pairs & ~c2_same_pairs)
    count_same_onlyin_c1 = np.sum(c1_same_pairs & ~c2_same_pairs)
    count_same_onlyin_c2 = np.sum(~c1_same_pairs & c2_same_pairs)
    
    rand_index = (count_same_in_both + count_different_in_both) / (count_same_in_both + count_same_onlyin_c1 + count_same_onlyin_c2 + count_different_in_both)
    jaccard_index = count_same_in_both / (count_same_in_both + count_same_onlyin_c1 + count_same_onlyin_c2)
    
    return rand_index, jaccard_index


def experiment_controller(file_path: str, distance: str = 'pattern_dtw', interClusterDistance: str = 'complete', 
                        transform: str = 'original', cMethod: str = 'inconsistent', cValue: float = 1.5,  replicate: int = 1, 
                        note: str = '', save_plots: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a controlled clustering experiment with comprehensive reporting.

    This function performs clustering experiments with different parameters and generates
    detailed reports including performance metrics, visualizations, and Excel summaries.

    Parameters
    ----------
    filePath : str
        Path to the input file containing time series data. Supports .xlsx and .csv formats.
        For .xlsx files, expects 'data' sheet with time series and optional 'clusters' sheet
        with cluster assignments.
    distance : str, default='pattern_dtw'
        Distance metric to use for clustering. Available options:
        
        **Pattern-based distances:**
        
        ``pattern``: Pattern distance using behavioral features
        
        ``pattern_dtw``: Pattern distance with Dynamic Time Warping

        ``dtw``: Dynamic Time Warping distance
        
        `Scipy distance metrics <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html>`_:

        ``euclidean``, ``minkowski``, ``cityblock``, ``seuclidean``, ``sqeuclidean``,
        ``cosine``, ``correlation``, ``hamming``, ``jaccard``, ``jensenshannon``,
        ``chebyshev``, ``canberra``, ``braycurtis``, ``mahalanobis``,
        ``yule``, ``matching``, ``dice``, ``rogerstanimoto``, ``russellrao``,
        ``sokalsneath``, ``kulczynski1``
        
    interClusterDistance : str, default='complete'
        `Hierarchical clustering linkage method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_:
        
        ``single``: Minimum distances between all observations of two sets

        ``complete``: Maximum distances between all observations of two sets

        ``average``: Average distances between all observations of two sets

        ``weighted``: Weighted average distances

        ``centroid``: Distance between centroids of clusters

        ``median``: Distance between medians of clusters

        ``ward``: Minimizes within-cluster sum of squared differences
        
    transform : str, default='original'
        Data transformation method applied before clustering:
        
        ``original``: No transformation applied

        ``normalize``: Min-max normalization to [0,1] range

        ``standardize``: Z-score standardization (mean=0, std=1)
        
    cMethod : str, default='maxclust'
        `Clustering criterion for forming flat clusters from hierarchy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_:
        
        ``maxclust``: Maximum number of clusters (requires cValue = number of clusters)

        ``distance``: Distance threshold (requires cValue = distance threshold)

        ``inconsistent``: Inconsistency criterion (requires cValue = inconsistency threshold)

        ``monocrit``: Monotonic criterion (requires cValue = threshold)
        
    cValue : float, default=1.5
        Threshold value for clustering criterion. Interpretation depends on cMethod:
        
        - For ``maxclust``: Number of desired clusters
        - For ``distance``: Distance threshold for cluster formation
        - For ``inconsistent``: Inconsistency coefficient threshold
        - For ``monocrit``: Threshold for monotonic criterion
        
    replicate : int, default=1
        Number of replications for pattern distance method. Only applicable when
        distance='pattern'. Higher values provide more robust results for
        stochastic pattern distance but increase computation time.
    note : str, default=''
        Additional note to append to output filename for identification purposes.
        Useful for experiment tracking and organization.
    save_plots : bool, default=False
        Whether to generate and save cluster visualization plots as PNG files.
        Plots show all time series grouped by their assigned clusters.
    output_dir : Optional[str], default=None
        Directory path to save output files. If None, creates 'output' directory 
        in the package root. Directory will be created if it doesn't exist.
        
    Returns
    -------
    Dict[str, Any]        
        - ``clusters``: Final cluster assignments as numpy array
        - ``cluster_list``: List of Cluster objects with detailed information
        - ``distance_matrix``: Computed condensed distance matrix  
        - ``rand_index``: Rand index comparing with original clusters (if available)
        - ``jaccard_index``: Jaccard index comparing with original clusters (if available)
        - ``run_time``: Clustering execution time in seconds
        - ``total_time``: Total experiment time including I/O in seconds
        - ``num_clusters``: Number of clusters formed
        - ``output_file``: Path to generated Excel report file
        - ``plot_file``: Path to generated plot file (if save_plots=True)
    """
    very_begin_time = time.time()
    
    list_of_ts_objects = read_time_series(file_path, withClusters=True)
    
    # Apply transformations
    if transform == 'normalize':
        list_of_ts_objects = _normalize_data(list_of_ts_objects)
    if transform == 'standardize':
        list_of_ts_objects = _standardize_data(list_of_ts_objects)

    
    # Perform clustering with replication for pattern distance method
    if distance == 'pattern':
        run_times = []
        rands = []
        jaccards = []
        
        for i in range(replicate):
            begin_time = time.time()
            try:
                dist_row, cluster_list, list_of_ts_objects = perform_clustering(list_of_ts_objects, distance, interClusterDistance, cMethod, cValue)
            except Exception as e:
                raise RuntimeError(f"Clustering failed on replication {i+1}: {e}")
                
            run_times.append(time.time() - begin_time)
            
            try:
                r, j = _compare_clusterings(list_of_ts_objects)
                rands.append(r)
                jaccards.append(j)
            except Exception:
                rands.append(None)
                jaccards.append(None)

        run_time = np.mean(run_times)
        rand = np.mean([r for r in rands if r is not None]) if any(r is not None for r in rands) else None
        jaccard = np.mean([j for j in jaccards if j is not None]) if any(j is not None for j in jaccards) else None
        
    else:
        # For other distance methods, no replication
        begin_time = time.time()
        try:
            dist_row, cluster_list, list_of_ts_objects = perform_clustering(list_of_ts_objects, distance, interClusterDistance, cMethod, cValue)
        except Exception as e:
            raise RuntimeError(f"Clustering failed: {e}")
            
        run_time = time.time() - begin_time

        try:
            rand, jaccard = _compare_clusterings(list_of_ts_objects)
        except Exception:
            rand, jaccard = None, None

    noClusters = max([each_cluster.cluster_id for each_cluster in cluster_list]) if cluster_list else 0
    
    outputFileName = f'{distance}-{interClusterDistance}-{transform}-{note}.xlsx'
    
    # Determine output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_path = os.path.join(output_dir, outputFileName)
    plot_file_path = None
    
    # Generate Excel report
    try:
        w = xlsxwriter.Workbook(output_file_path)
        ws = w.add_worksheet('results')
        ws.set_column('A:A', 20)
        ws.set_column('B:B', 14)
        ws.set_column('C:C', 14)
        
        # Write header information
        ws.write(0, 0, 'File Path:')
        ws.write(0, 1, file_path)
        ws.write(1, 0, 'Inter-Cluster Similarity')
        ws.write(1, 1, interClusterDistance)
        ws.write(2, 0, 'cMethod')
        ws.write(2, 1, cMethod)
        ws.write(3, 0, 'cValue')
        ws.write(3, 1, cValue)
        ws.write(4, 0, "Time:")
        ws.write(4, 1, time.strftime("%H:%M %d/%m/%Y"))
        
        # Write distance method information
        ws.write(5, 0, 'Distance Measure:')
        ws.write(6, 0, distance)
        ws.write(7, 0, distance)
        ws.write(8, 0, distance)
        
        # Write metrics
        ws.write(5, 2, 'Metric')
        ws.write(6, 2, 'Jaccard')
        ws.write(7, 2, 'Rand')
        ws.write(8, 2, 'Run Time')
        
        # Write transformation information
        ws.write(5, 4, 'Transformation')
        ws.write(6, 4, transform)
        ws.write(7, 4, transform)
        ws.write(8, 4, transform)
        
        # Write outcomes
        ws.write(5, 5, 'Outcome')
        ws.write(6, 5, jaccard)
        ws.write(7, 5, rand)
        ws.write(8, 5, run_time)
        
        # Write cluster information
        ws.write(11, 0, 'Total number of clusters:')
        ws.write(11, 1, noClusters)

        for each_cluster in cluster_list:
            ws.write(11+each_cluster.cluster_id, 0, f'Cluster {each_cluster.cluster_id}:')
            ws.write(11+each_cluster.cluster_id, 1, each_cluster.number_of_members)
        
        # Write individual cluster assignments
        start_row = 23
        ws.write(start_row, 0, 'Index')
        ws.write(start_row, 1, 'Original Cluster')
        ws.write(start_row, 2, 'Cluster List')
        
        for each_ts in list_of_ts_objects:
            ws.write(start_row + 1 + each_ts.index, 0, each_ts.index)
            ws.write(start_row + 1 + each_ts.index, 1, each_ts.previous_cluster_id)
            ws.write(start_row + 1 + each_ts.index, 2, each_ts.cluster_id)
            
        w.close()
        print(f"Excel report saved to: {output_file_path}")
        
    except Exception as e:
        warnings.warn(f"Failed to generate Excel report: {e}")
        output_file_path = None
    
    # Generate plots if requested
    if save_plots:
        try:
            plot_base_name = os.path.splitext(outputFileName)[0]
            plot_file_path = os.path.join(output_dir, plot_base_name)
            plot_clusters(cluster_list, distance, mode='save', fname=plot_file_path)
            print(f"Cluster plots saved to: {plot_file_path}.png")
        except Exception as e:
            warnings.warn(f"Failed to generate plots: {e}")
            plot_file_path = None
    
    very_end_time = time.time()
    total_time = very_end_time - very_begin_time
    print(f'Total experiment time: {total_time:.2f} seconds')
    
    results = {
        'cluster_list': cluster_list,
        'list_of_ts_objects': list_of_ts_objects,
        'distance_matrix': dist_row,
        'rand_index': rand,
        'jaccard_index': jaccard,
        'run_time': run_time,
        'total_time': total_time,
        'num_clusters': noClusters,
        'output_file': output_file_path,
        'plot_file': plot_file_path
    }
    
    return results