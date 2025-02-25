'''

Created on Nov 8, 2011
Updated on December 15, 2014

.. codeauthor:: 
     gyucel <gonenc.yucel (at) boun (dot) edu (dot) tr>,
'''
from __future__ import division
import numpy as np
from behavior_splitter import construct_features 


def dtw_distance(d1, d2, wSlopeError, wCurvatureError):
    '''
    Calculates the distance between two feature vectors using the Dynamic Time Warping method. Returns the avg. distance per data section.
    In other words, avg_dist = dtw_dist / warping_path_length
    
    :param series1: Feature vector 1 (2-dimensional numpy array).
    :param series2: Feature vector 2 (2-dimensional numpy array).
    :param wDim1: Weight of the error between the 1st dimensions of the two 
                  feature vectors (i.e. Slope).
    :param wDim2: Weight of the error between the 2nd dimensions of the two 
                  feature vectors (i.e. Curvature).
    '''

    dtw = np.zeros([d1.shape[1] + 1, d2.shape[1] + 1])
    dtw[:, 0] = np.inf
    dtw[0, :] = np.inf
    dtw[0, 0] = 0
    for i in range(d1.shape[1]):
        for j in range(d2.shape[1]):
            cost = local_dist(d1[:,i], d2[:,j], wSlopeError, wCurvatureError)
            dtw[i + 1, j + 1] = cost + min([dtw[i + 1, j], dtw[i, j + 1], dtw[i, j]])
    
    i = d1.shape[1] 
    j = d2.shape[1]
    w_path = 0
    while (i>0 and j>0):
        if i==0:
            j = j-1
            w_path = w_path+1
        elif j==0:
            i=i-1
            w_path = w_path+1
        else:
            if dtw[i-1,j] == min(dtw[i-1,j-1],dtw[i,j-1],dtw[i-1,j]):
                i = i-1
                w_path = w_path+1
            elif dtw[i,j-1] == min(dtw[i-1,j-1],dtw[i,j-1],dtw[i-1,j]):
                j=j-1
                w_path = w_path+1
            else:
                i=i-1
                j=j-1
                w_path=w_path+1
    
    return dtw[d1.shape[1], d2.shape[1]]/w_path

    
def local_dist(x, y, wDim1=1, wDim2=1):
    error = np.square(x - y)
    weights = np.array([wDim1, wDim2])
    error = error*weights[np.newaxis, np.newaxis, :]
    # this is sort of stupid, it should be possible to do in one line
    # but axis=(1,2) does not work, it requires numpy 1.7. I have 1.6.1 
    # at the moment
    error = np.sum(error, axis=1)
    error = np.sum(error, axis=1)
    return error


def distance_pattern_dtw(data, significanceLevel=0.01, 
                    wSlopeError=1, 
                    wCurvatureError=1):
    
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
    
    
    data_w_desc = []
    #Generates the feature vectors for all the time series that are contained 
    # in numpy array data
    features = construct_features(data, significanceLevel)
    dRow = np.zeros(shape=(np.sum(np.arange(len(data))), ))
    
    index = -1
    for i in range(len(data)):
        feature_i = features[i]
            
        # For each timeseries data, a behavior descriptor dictionary is created.
        # This has key information for post-clustering analysis
        # These descriptors are stored with the original data in a global array named data_w_desc
        behaviorDesc = {}
        behaviorDesc['Index'] = str(i)
        
        #this may not work due to data type mismatch
        featVector = feature_i
        
        behaviorDesc['Feature vector'] = str(featVector)
        behavior = data[i]
        localLog = (behaviorDesc, behavior)
        data_w_desc.append(localLog)
    
        for j in range(i+1, len(data)):
            index += 1
            feature_j = features[j]
            distance = dtw_distance(feature_i, feature_j, wSlopeError, wCurvatureError)
            dRow[index] = distance
    return dRow, data_w_desc


if __name__ == '__main__':
    print 'deneme'
    