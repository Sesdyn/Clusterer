import numpy as np
from numba import njit, prange
from typing import Tuple, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from simclstr.clusterer import TimeSeries

def _distance_dtw(list_of_ts_objects: List['TimeSeries'], metric: str = 'dtw', distance_kwargs: dict = {}) -> Tuple[np.ndarray, List['TimeSeries']]:
    """
    Calculate pairwise Dynamic Time Warping (DTW) distances between all data sequences.

    Dynamic Time Warping is a technique for measuring similarity between two temporal
    sequences that may vary in speed or timing. Unlike Euclidean distance, DTW can
    handle sequences of different lengths and finds the optimal alignment between them
    by allowing stretching and compression of the time axis.
    
    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects.
    metric: str, default='dtw'
    distance_kwargs : dict, default={}
        Distance dtw does not need any additional parameters.

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
        Each element represents the DTW distance between a pair of sequences.
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects with updated feature vector.

    This implementation uses absolute difference as the local distance measure
    between individual points: |x_i - y_j|.
    """
    # For distance_dtw, the feature vector is the data itself
    for i, each_ts in enumerate(list_of_ts_objects):
        each_ts.feature_vector = each_ts.data

    # Convert list of arrays to 2D numpy array for distance functions
    data = np.array([ts.data for ts in list_of_ts_objects])

    n = len(data)
    dRow = np.zeros(shape=(n * (n - 1) // 2,))

    dRow = compute_dtw_distances(data, dRow)

    return dRow, list_of_ts_objects

@njit(parallel=True)
def compute_dtw_distances(data: np.ndarray, dRow: np.ndarray) -> np.ndarray:
    """
    Compute DTW distances between all pairs using Numba for performance.

    Parameters
    ----------
    data : np.ndarray
        Input sequences array.
    dRow : np.ndarray
        Array to store distances.

    Returns
    -------
    np.ndarray
        Array filled with DTW distances.
    """
    n = len(data)
    for i in prange(n - 1):
        base = i * n - (i * (i + 1)) // 2
        sample1 = data[i]
        n1 = sample1.shape[0]
        for j in range(i + 1, n):
            index = base + (j - i - 1)
            sample2 = data[j]
            n2 = sample2.shape[0]

            dtw = np.full((n1 + 1, n2 + 1), np.inf)
            dtw[0, 0] = 0

            for k in range(n1):
                for l in range(n2):
                    cost = abs(sample1[k] - sample2[l])
                    dtw[k + 1, l + 1] = cost + min(dtw[k + 1, l], dtw[k, l + 1], dtw[k, l])

            dRow[index] = dtw[n1, n2]
            
    return dRow