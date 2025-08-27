import numpy as np
from typing import Tuple, List, TYPE_CHECKING
from ._behavior_splitter import _construct_features

if TYPE_CHECKING:
    from simclstr.clusterer import TimeSeries

def distance_same_length(series1: np.ndarray, series2: np.ndarray, wDim1: float, wDim2: float) -> float:
    """
    Calculate the distance between two feature vectors of the same size.

    Parameters
    ----------
    series1 : np.ndarray
        Feature vector 1 with shape (n_sections, 2) where [:, 0] is slope and [:, 1] is curvature.
    series2 : np.ndarray
        Feature vector 2 with shape (n_sections, 2) where [:, 0] is slope and [:, 1] is curvature.
    wDim1 : float
        Weight of the error between the slope dimensions ([:, 0]) of the two feature vectors.
    wDim2 : float
        Weight of the error between the curvature dimensions ([:, 1]) of the two feature vectors.

    Returns
    -------
    float
        Weighted distance between the feature vectors.
    """
    diff_sq = np.square(series1 - series2)
    weighted_error = wDim1 * diff_sq[:, 0] + wDim2 * diff_sq[:, 1]
    return np.sum(weighted_error) / series1.shape[0]


def distance_different_length(series1: np.ndarray, series2: np.ndarray, wDim1: float, wDim2: float, sisterCount: int) -> float:
    """
    Calculate the distance between two feature vectors of different sizes.

    Parameters
    ----------
    series1 : np.ndarray
        Feature vector 1 with shape (n_sections, 2) where [:, 0] is slope and [:, 1] is curvature.
    series2 : np.ndarray
        Feature vector 2 with shape (n_sections, 2) where [:, 0] is slope and [:, 1] is curvature.
    wDim1 : float
        Weight of the error between the slope dimensions ([:, 0]) of the two feature vectors.
    wDim2 : float
        Weight of the error between the curvature dimensions ([:, 1]) of the two feature vectors.
    sisterCount : int
        Number of long-versions that will be created for the short vector.

    Returns
    -------
    float
        Minimum distance among all generated sister vectors.
    """
    length1 = series1.shape[0]
    length2 = series2.shape[0]

    if length1 > length2:
        shortFV = series2
        longFV = series1
    else:
        shortFV = series1
        longFV = series2

    sisters = create_sisters(shortFV, (longFV.shape[0], 2), sisterCount)
    error = np.square(sisters - longFV[np.newaxis, :, :])
    weights = np.array([wDim1, wDim2], dtype=np.float64)

    error = error * weights[np.newaxis, np.newaxis, :]
    total_error = np.sum(error, axis=(1, 2))

    return np.min(total_error) / longFV.shape[0]


def create_sisters(shortFV: np.ndarray, desired_shape: Tuple[int, int], sister_count: int) -> np.ndarray:
    """    
    Creates a set of extended feature vectors by randomly sampling and reordering
    sections from the short feature vector (shortFV) to match the desired length.

    Parameters
    ----------
    shortFV : np.ndarray
        The feature vector to be extended with shape (n_sections, 2).
    desired_shape : Tuple[int, int]
        The desired shape (n_sections, 2) of the extended feature vectors (i.e. sisters).
    sister_count : int
        The desired number of sisters to be created.

    Returns
    -------
    np.ndarray
        Array of sister vectors with shape (sister_count, desired_shape[0], 2).
    """
    to_add = desired_shape[0] - shortFV.shape[0]
    short_length = shortFV.shape[0]
    
    indices = np.empty((sister_count, desired_shape[0]), dtype=np.int32)
    
    random_indices = np.random.randint(0, short_length, size=(sister_count, to_add))
    
    base_indices = np.arange(short_length)
    
    indices[:, :to_add] = random_indices
    indices[:, to_add:] = base_indices[np.newaxis, :]

    indices.sort(axis=1)
    sisters = shortFV[indices, :] 

    return sisters


def _distance_pattern(list_of_ts_objects: List['TimeSeries'], metric: str = 'pattern', distance_kwargs: dict = {}) -> Tuple[np.ndarray, List['TimeSeries']]:
    """
    This function computes distances based on qualitative behavioral patterns rather than
    raw data values. It extracts slope and curvature features from time series and compares
    these behavioral characteristics to determine similarity between different dynamic modes.
    
    The pattern distance is particularly useful for identifying similar behavioral patterns
    even when the actual values differ significantly, focusing on the shape and trends
    rather than magnitude.

    Parameters
    ----------
    list_of_ts_objects : List['TimeSeries']
        List of TimeSeries objects.
    metric: str, default='pattern'
    distance_kwargs : dict, default={}

        Dictionary of keyword arguments for the distance calculation:

        ``significanceLevel`` : float, default=0.0001
            Threshold value (as a fraction) for filtering out insignificant fluctuations in 
            slope and curvature calculations. Values below this threshold relative to the
            data magnitude are considered noise and set to zero.
        ``sisterCount`` : int, default=50
            Number of extended versions created for shorter feature vectors when comparing
            sequences with different behavioral segment counts. Higher values provide more
            thorough comparison but increase computation time.
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
    sisterCount = distance_kwargs.get('sisterCount', 50)
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

    index = 0
    for i in range(n):
        feature_i = transposed_features[i]
        for j in range(i+1, n):
            feature_j = transposed_features[j]

            if feature_i.shape[0] == feature_j.shape[0]:
                distance = distance_same_length(feature_i, feature_j, wSlopeError, wCurvatureError)
            else:
                distance = distance_different_length(feature_i, feature_j, wSlopeError, wCurvatureError, sisterCount)

            dRow[index] = distance
            index += 1

    return dRow, list_of_ts_objects