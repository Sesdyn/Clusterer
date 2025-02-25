'''

Created on Nov 8, 2011
Updated on Dec 15, 2014

.. codeauthor:: 
     gyucel <gonenc.yucel (at) boun (dot) edu (dot) tr>
'''
from __future__ import division
import random
import logging as log
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
    
    error = np.square(series1-series2)   
    error = np.array([wDim1, wDim2])[np.newaxis].T * error
    error = np.sum(error)
    return error/series1.shape[1]

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
    
    error = 10000000
    
    length1 = series1.shape[1]
    length2 = series2.shape[1]
    toAdd = abs(length1-length2)
    
    if length1>length2:
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
    error = np.square(sisters - longFV.T[np.newaxis,:,:])
    weights = np.array([wDim1, wDim2])
    
    error = error*weights[np.newaxis, np.newaxis, :]
    
    # this is sort of stupid, it should be possible to do in one line
    # but axis=(1,2) does not work, it requires numpy 1.7. I have 1.6.1 
    # at the moment
    error = np.sum(error, axis=1)
    error = np.sum(error, axis=1)
    return np.min(error)/longFV.shape[1]


def create_sisters(shortFV, desired_shape, sister_count):
    '''
    Creates a set of new feature vectors that are behaviorally identical to the given 
    short feature vector (shortFV), and that have the stated number of segments (i.e. desired_shape).
    
    :param shortFV: The feature vector to be extended.
    :param desired_shape: The desired shape (2-by-number of sections) of the extended feature vectors (i.e. sisters) 
    :param sister_count: The desired number of sisters to be created
    ''' 
    
    #determine how much longer the vector has to become
    to_add = desired_shape[1]-shortFV.shape[1]
    
    #create a 2d array of indices
    indices = np.zeros(shape=(sister_count, desired_shape[1]),dtype=int)
    
    #fill the first part of the indices array with random numbers
    #these are the indices that will be used to extent the short vector
    indices[:, 0:to_add] = np.random.randint(0, 
                                             shortFV.shape[1], 
                                             size=(sister_count, to_add))
    
    #add the indices for the full vector to the rest
    indices[:, to_add::] = np.arange(0, shortFV.shape[1])[np.newaxis, :]
    
    #sort indices
    indices = np.sort(indices, axis=1)
    
    #this is where the real magic happens, we use the generated indices
    #in order to generate in one line of code all the sisters
    sisters = shortFV.T[indices,:] 
    
    return sisters


def createSister(shortFV, toAdd):
    '''
    Creates a new feature vector that is behaviorally identical to the given 
    vector by adding toAdd number of segments.
    
    :param shortFV: The feature vector to be extended.
    :param toAdd: Number of sections to be added to the input vector while 
                  creating the equivalent sister.
    ''' 
    sister = np.zeros(shape=(shortFV.shape[0], shortFV.shape[1]+toAdd))
    
    
    # while you have to add, add a random number of entries
    index = 0 
    i = 0
    while (toAdd>0) and (i<shortFV.shape[1]):
        
        x = random.randint(0, toAdd)
        sister[:, index:index+x+1] = shortFV[:,i][np.newaxis].T
        toAdd -= x
        index += x+1
        i+=1
    
    #fill up with the remaining values from short
    if i<shortFV.shape[1]:
        sister[:, index::] = shortFV[:, i::]

    return sister

def distance_pattern(data,significanceLevel=0.01,
                    sisterCount=50, 
                    wSlopeError=1, 
                    wCurvatureError=1,
                    ):
    
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
    
    
    data_w_desc = []
    #Generates the feature vectors for all the time series that are contained 
    # in numpy array data
    features = construct_features(data, significanceLevel)
    log.info("calculating distances")
    dRow = np.zeros(shape=(np.sum(np.arange(len(data))), ))
    #print dRow.shape
    
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
            if feature_i.shape[1] == feature_j.shape[1]:
                distance = distance_same_length(feature_i, feature_j, 
                                                wSlopeError, wCurvatureError)
    
            else:
                distance = distance_different_lenght(feature_i, 
                                                     feature_j, 
                                                     wSlopeError, 
                                                     wCurvatureError, 
                                                     sisterCount)
            dRow[index] = distance
    #print data_w_desc.__class__
    return dRow, data_w_desc


if __name__ == '__main__':
    print 'deneme'
    