import numpy as np
from numba import njit
from typing import Tuple, List, TYPE_CHECKING
from ._behavior_splitter import _construct_features

if TYPE_CHECKING:
    from simclstr.clusterer import TimeSeries

def _distance_pattern_dtw(list_of_ts_objects: List['TimeSeries'], metric: str = 'pattern_dtw', distance_kwargs: dict = {}) -> Tuple[np.ndarray, List['TimeSeries']]:
    """
    Calculate pairwise pattern distances between all data sequences using Dynamic Time Warping.
    
    The distance measures the proximity of data series in terms of their 
    qualitative pattern features. In other words, it quantifies the proximity 
    between two different dynamic behaviour modes by allowing for time warping.

    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects.
    metric: str, default='pattern_dtw'
    distance_kwargs : dict, default={}

        Dictionary of keyword arguments for the distance calculation:

        ``significanceLevel`` : float, default=0.0001
            Threshold value (as a fraction) for filtering out insignificant fluctuations in 
            slope and curvature calculations. Values below this threshold relative to the
            data magnitude are considered noise and set to zero.
        ``wSlopeError`` : float, default=1.0
            Weight applied to slope dimension errors when calculating feature vector distances.
        ``wCurvatureError`` : float, default=1.0
            Weight applied to curvature dimension errors when calculating feature vector distances.

    Returns
    -------
    dRow : np.ndarray
        Condensed distance matrix as 1D array of length n_samples * (n_samples - 1) / 2.
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects with updated index and feature vector.
    """
    significanceLevel = distance_kwargs.get('significanceLevel', 0.0001)
    wSlopeError = distance_kwargs.get('wSlopeError', 1)
    wCurvatureError = distance_kwargs.get('wCurvatureError', 1)

    # Convert list of arrays to 2D numpy array for distance functions
    data = np.array([ts.data for ts in list_of_ts_objects])

    features = _construct_features(data, significanceLevel)

    transposed_features = [feature.T for feature in features]

    n = len(data)
    dRow = np.zeros(shape=(n * (n - 1) // 2,))

    # Update the feature vector and index of the TimeSeries objects
    for i, each_ts in enumerate(list_of_ts_objects):
        each_ts.feature_vector = transposed_features[i]
        each_ts.index = i

    dRow = _compute_pattern_dtw_distances(features, dRow, wSlopeError, wCurvatureError)

    return dRow, list_of_ts_objects


@njit
def _compute_pattern_dtw_distances(features: List[np.ndarray], dRow: np.ndarray, wSlopeError: float, wCurvatureError: float) -> np.ndarray:
    """
    Compute pairwise DTW distances between all feature vectors.

    Parameters
    ----------
    features : List[np.ndarray]
        List of feature vectors, each with shape (2, n_sections).
    dRow : np.ndarray
        Pre-allocated array to store the condensed distance matrix.
    wSlopeError : float
        Weight for the slope dimension error.
    wCurvatureError : float
        Weight for the curvature dimension error.

    Returns
    -------
    np.ndarray
        Updated distance array containing pairwise DTW distances.
    """
    index = 0
    for i in range(len(features)):
        feature_i = features[i]
        for j in range(i + 1, len(features)):
            feature_j = features[j]
            distance = _dtw_distance(feature_i, feature_j, wSlopeError, wCurvatureError)
            dRow[index] = distance
            index += 1
    return dRow


@njit
def _dtw_distance(d1: np.ndarray, d2: np.ndarray, wSlopeError: float, wCurvatureError: float) -> float:
    """
    Calculate the distance between two feature vectors using Dynamic Time Warping.

    Parameters
    ----------
    d1 : np.ndarray
        Feature vector 1 with shape (2, n_sections) where [0, :] is slope and [1, :] is curvature.
    d2 : np.ndarray
        Feature vector 2 with shape (2, n_sections) where [0, :] is slope and [1, :] is curvature.
    wSlopeError : float
        Weight of the error between the slope dimensions (first dimension) of the two feature vectors.
    wCurvatureError : float
        Weight of the error between the curvature dimensions (second dimension) of the two feature vectors.
        
    Returns
    -------
    float
        Average DTW distance per data section.
    """
    n1, n2 = d1.shape[1], d2.shape[1]
    dtw = np.full((n1 + 1, n2 + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(n1):
        for j in range(n2):
            slope_diff = d1[0, i] - d2[0, j]
            curv_diff = d1[1, i] - d2[1, j]
            cost = wSlopeError * (slope_diff * slope_diff) + wCurvatureError * (curv_diff * curv_diff)
            dtw[i + 1, j + 1] = cost + min(dtw[i + 1, j], dtw[i, j + 1], dtw[i, j])

    i, j = n1, n2
    w_path = 0
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            w_path += 1
        elif j == 0:
            i -= 1
            w_path += 1
        else:
            up = dtw[i-1, j]
            left = dtw[i, j-1]
            diag = dtw[i-1, j-1]
            if up <= left and up <= diag:
                i -= 1
                w_path += 1
            elif left <= up and left <= diag:
                j -= 1
                w_path += 1
            else:
                i -= 1
                j -= 1
                w_path += 1
    
    return dtw[n1, n2] / w_path