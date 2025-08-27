import numpy as np
from numba import njit
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
        List of TimeSeries objects with updated index and feature vector.

    This implementation uses absolute difference as the local distance measure
    between individual points: |x_i - y_j|.
    """
    # For distance_dtw, the feature vector is the data itself
    for i, each_ts in enumerate(list_of_ts_objects):
        each_ts.feature_vector = each_ts.data
        each_ts.index = i

    # Convert list of arrays to 2D numpy array for distance functions
    data = np.array([ts.data for ts in list_of_ts_objects])

    dRow = np.zeros(shape=(np.sum(np.arange(len(data))), ))

    dRow = compute_dtw_distances(data, dRow)

    return dRow, list_of_ts_objects

@njit
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
    index = -1
    for i in range(len(data)):            
        for j in range(i+1, len(data)):
            index += 1
            
            sample1 = data[i]
            sample2 = data[j]
            
            dtw = np.zeros((sample1.shape[0] + 1, sample2.shape[0] + 1))
            dtw[:, 0] = np.inf
            dtw[0, :] = np.inf
            dtw[0, 0] = 0
            
            for k in range(sample1.shape[0]):
                for l in range(sample2.shape[0]):
                    cost = np.absolute(sample1[k] - sample2[l])
                    dtw[k + 1, l + 1] = cost + min(dtw[k + 1, l], dtw[k, l + 1], dtw[k, l])
            
            distance = dtw[sample1.shape[0], sample2.shape[0]]
            dRow[index] = distance
            
    return dRow