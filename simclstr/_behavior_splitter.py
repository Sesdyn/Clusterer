import numpy as np
from typing import List

def _construct_features(data: np.ndarray, significanceLevel: float = 0.01) -> List[np.ndarray]:
    """Construct feature vectors for time series data based on behavioral characteristics.

    Parameters
    ----------
    data : np.ndarray
        Input data as 2D array. Each row represents a time series.
    significanceLevel : float, optional
        Threshold level for determining significant changes. Default is 0.01.

    Returns
    -------
    List[np.ndarray]
        A list where each element is a 2D array with shape (2, n_sections)
        containing slope and curvature signs for that series. The first row is
        slope sign, the second row is curvature sign. Lengths can vary by series.
    """
    features = []

    slopes = np.gradient(data, axis=1)
    curvatures = np.gradient(slopes, axis=1)

    for i in range(data.shape[0]):
        feature = _split_behavior(data[i], slopes[i], curvatures[i], significanceLevel)
        features.append(feature)

    return features


def _split_behavior(data_series: np.ndarray, slope: np.ndarray, curvature: np.ndarray, significanceLevel: float = 0.01) -> np.ndarray:
    """Split a time series into sections of different atomic behavior modes.

    Parameters
    ----------
    data_series : np.ndarray
        1D array representing a single time series.
    slope : np.ndarray
        1D array containing the first derivative.
    curvature : np.ndarray
        1D array containing the second derivative.
    significanceLevel : float, optional
        Threshold level for filtering out insignificant changes.

    Returns
    -------
    np.ndarray
        2D array with shape (2, n_sections) containing slope and curvature signs.
        The first dimension (2) represents: [0] slope sign, [1] curvature sign.
    """
    abs_data = np.abs(data_series)
    abs_slope = np.abs(slope)
    abs_curvature = np.abs(curvature)
    
    data_threshold = abs_data * significanceLevel
    slope_threshold = abs_slope * significanceLevel

    slope = slope * (abs_slope >= data_threshold)
    curvature = curvature * (abs_curvature >= slope_threshold)

    sign_slope = np.sign(slope)
    sign_curvature = np.sign(curvature)

    sections = sign_slope * 10 + sign_curvature

    transitions = np.diff(sections)
    transition_points = np.flatnonzero(transitions)
    number_of_sections = len(transition_points) + 1

    feature_vector = np.empty((2, number_of_sections))

    if number_of_sections == 1:
        feature_vector[0, 0] = sign_slope[0]
        feature_vector[1, 0] = sign_curvature[0]
    else:
        section_starts = np.concatenate(([0], transition_points + 1))
        feature_vector[0] = sign_slope[section_starts]
        feature_vector[1] = sign_curvature[section_starts]

    return feature_vector