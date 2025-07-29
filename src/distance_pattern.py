'''
Created on Nov 8, 2011
Updated on Dec 15, 2014
gyucel <gonenc.yucel (at) boun (dot) edu (dot) tr>

Updated July 25, 2025
'''

import numpy as np
from behavior_splitter import construct_features


def distance_same_length(series1, series2, wDim1, wDim2):
    '''
    Calculates the distance between two feature vectors of the same size.
    
    :param series1: Feature vector 1 (2-dimensional numpy array).
    :param series2: Feature vector 2 (2-dimensional numpy array).
    :param wDim1: Weight of the error between the 1st dimensions of the two 
                  feature vectors (i.e. Slope).
    :param wDim2: Weight of the error between the 2nd dimensions of the two 
                  feature vectors (i.e. Curvature).
    '''
    
    diff_sq = np.square(series1 - series2)
    weighted_error = wDim1 * diff_sq[0] + wDim2 * diff_sq[1]
    return np.sum(weighted_error) / series1.shape[1]


def distance_different_lenght(series1, series2, wDim1, wDim2, sisterCount):
    '''
    Calculates the distance between two feature vectors of different sizes.
    
    :param series1: Feature vector 1 (2-dimensional numpy array).
    :param series2: Feature vector 2 (2-dimensional numpy array).
    :param wDim1: Weight of the error between the 1st dimensions of the two 
                  feature vectors (i.e. Slope).
    :param wDim2: Weight of the error between the 2nd dimensions of the two 
                  feature vectors (i.e. Curvature).
    :param sisterCount: Number of long-versions that will be created for the 
                        short vector.
    '''
    
    length1 = series1.shape[1]
    length2 = series2.shape[1]
    
    if length1 > length2:
        shortFV = series2
        longFV = series1
    else:
        shortFV = series1
        longFV = series2

    sisters = create_sisters(shortFV, longFV.shape, sisterCount)

    # to take advantage of the fact that the sisters are in a 3d array
    # I also vectorized the error calculation.
    # this means that calculation time is almost independent from the number
    # of sisters you want to use.
    error = np.square(sisters - longFV.T[np.newaxis, :, :])
    weights = np.array([wDim1, wDim2], dtype=np.float64)

    error = error * weights[np.newaxis, np.newaxis, :]

    total_error = np.sum(error, axis=(1, 2))

    return np.min(total_error) / longFV.shape[1]


def create_sisters(shortFV, desired_shape, sister_count):
    '''
    Creates a set of new feature vectors that are behaviorally identical to the given 
    short feature vector (shortFV), and that have the stated number of segments (i.e. desired_shape).
    
    :param shortFV: The feature vector to be extended.
    :param desired_shape: The desired shape (2-by-number of sections) of the extended feature vectors (i.e. sisters) 
    :param sister_count: The desired number of sisters to be created
    ''' 
    
    # Determine how much longer the vector has to become
    to_add = desired_shape[1] - shortFV.shape[1]
    short_length = shortFV.shape[1]
    
    #create a 2d array of indices
    indices = np.zeros(shape=(sister_count, desired_shape[1]),dtype=int)
    
    indices = np.empty((sister_count, desired_shape[1]), dtype=np.int32)
    
    # Generate all random indices at once
    random_indices = np.random.randint(0, short_length, size=(sister_count, to_add))
    
    # Create base indices for the original vector
    base_indices = np.arange(short_length)
    
    indices[:, :to_add] = random_indices
    indices[:, to_add:] = base_indices[np.newaxis, :]

    # Sort indices to maintain order
    indices.sort(axis=1)
    
    #this is where the real magic happens, we use the generated indices
    #in order to generate in one line of code all the sisters
    sisters = shortFV.T[indices,:] 

    return sisters


def distance_pattern(data, significanceLevel=0.01, sisterCount=50, wSlopeError=1, wCurvatureError=1):
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
    :param sisterCount: Number of long-versions that will be created for the 
                        short vector while comparing two data series with 
                        unequal feature vector lengths. 
    :param wSlopeError: Weight of the error between the 1st dimensions of the 
                        two feature vectors (i.e. Slope). (default=1)
    :param wCurvatureError: Weight of the error between the 2nd dimensions of 
                            the two feature vectors (i.e. Curvature). 
                            (default=1)
    '''

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Generate feature vectors for all time series that are contained
    features = construct_features(data, significanceLevel)
    #log.info("calculating distances")

    n = len(data)
    dRow = np.zeros(shape=(n * (n - 1) // 2,))

    data_w_desc = [({'Index': str(i), 'Feature vector': str(features[i])}, data[i]) 
                   for i in range(n)]

    index = 0
    for i in range(n):
        feature_i = features[i]
        for j in range(i+1, n):
            feature_j = features[j]

            if feature_i.shape[1] == feature_j.shape[1]:
                distance = distance_same_length(feature_i, feature_j, wSlopeError, wCurvatureError)
            else:
                distance = distance_different_lenght(feature_i, feature_j, wSlopeError, wCurvatureError, sisterCount)

            dRow[index] = distance
            index += 1

    return dRow, data_w_desc


if __name__ == '__main__':
    print('gonenc')