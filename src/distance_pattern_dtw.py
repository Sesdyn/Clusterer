'''
Created on Nov 8, 2011
Updated on December 15, 2014
gyucel <gonenc.yucel (at) boun (dot) edu (dot) tr>

Updated July 25, 2025
'''
from __future__ import division
import numpy as np
from numba import njit
from behavior_splitter import construct_features


@njit
def dtw_distance_numba(d1, d2, wSlopeError, wCurvatureError):
    '''
    Calculates the distance between two feature vectors using the Dynamic Time Warping method. Returns the avg. distance per data section.
    In other words, avg_dist = dtw_dist / warping_path_length
    
    :param series1: Feature vector 1 (2-dimensional numpy array).
    :param series2: Feature vector 2 (2-dimensional numpy array).
    :param wSlopeError: Weight of the error between the 1st dimensions of the two 
                  feature vectors (i.e. Slope).
    :param wCurvatureError: Weight of the error between the 2nd dimensions of the two 
                  feature vectors (i.e. Curvature).
    '''
    n1, n2 = d1.shape[1], d2.shape[1]
    dtw = np.full((n1 + 1, n2 + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(n1):
        for j in range(n2):
            cost = local_dist_numba(d1[:, i], d2[:, j], wSlopeError, wCurvatureError)
            dtw[i + 1, j + 1] = cost + min(dtw[i + 1, j], dtw[i, j + 1], dtw[i, j])

    # Backtrack to find path length
    i, j = n1, n2
    w_path = 0
    while i > 0 and j > 0:
        if i == 0:
            j -= 1
            w_path += 1
        elif j == 0:
            i -= 1
            w_path += 1
        else:
            min_val = min(dtw[i-1, j-1], dtw[i, j-1], dtw[i-1, j])
            if dtw[i-1, j] == min_val:
                i -= 1
                w_path += 1
            elif dtw[i, j-1] == min_val:
                j -= 1
                w_path += 1
            else:
                i -= 1
                j -= 1
                w_path += 1
    
    return dtw[n1, n2] / w_path


@njit
def local_dist_numba(x, y, wDim1=1, wDim2=1):
    error = np.square(x - y)
    return wDim1 * error[0] + wDim2 * error[1]


def distance_pattern_dtw(data, significanceLevel=0.01, wSlopeError=1, wCurvatureError=1):
    
    '''
    The distance measures the proximity of data series in terms of their 
    qualitative pattern features. In order words, it quantifies the proximity 
    between two different dynamic behaviour modes.
    
    It is designed to work mainly on non-stationary data. It's current version 
    does not perform well in catching the proximity of two cyclic/repetitive 
    patterns with different number of cycles (e.g. oscillation with 4 cycle 
    versus oscillation with 6 cycles).
    
    :param significanceLevel:  The threshold value to be used in filtering out 
                               fluctuations in the slope and the curvature. (default=0.01)
    :param wSlopeError: Weight of the error between the 1st dimensions of the 
                        two feature vectors (i.e. Slope). (default=1)
    :param wCurvatureError: Weight of the error between the 2nd dimensions of 
                            the two feature vectors (i.e. Curvature). 
                            (default=1)
    '''
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Generate feature vectors for all time series
    features = construct_features(data, significanceLevel)
    
    n = len(data)
    dRow = np.zeros(shape=(n * (n - 1) // 2,))
    
    # Create behavior descriptors
    data_w_desc = [({'Index': str(i), 'Feature vector': str(features[i])}, data[i]) for i in range(n)]
    
    dRow = compute_pattern_dtw_distances_numba(features, dRow, wSlopeError, wCurvatureError)
    
    return dRow, data_w_desc


@njit
def compute_pattern_dtw_distances_numba(features, dRow, wSlopeError, wCurvatureError):
    index = 0
    for i in range(len(features)):
        feature_i = features[i]
        for j in range(i + 1, len(features)):
            feature_j = features[j]
            distance = dtw_distance_numba(feature_i, feature_j, wSlopeError, wCurvatureError)
            dRow[index] = distance
            index += 1
    return dRow


if __name__ == '__main__':
    print('gonenc')