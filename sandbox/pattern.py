'''

Created on Nov 8, 2011
Updated on March 26, 2012

.. codeauthor:: 
     gyucel <g.yucel (at) tudelft (dot) nl>,
     jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division
import random
import logging as log
import numpy as np



def distance_pattern(sample1, sample2, wSlopeError, wCurvatureError):
    
    sisterCount = 50
    
    if sample1.shape[0] == sample2.shape[0]:
        distance = distance_same_length(sample1, sample2, wSlopeError, wCurvatureError)
    else:
        distance = distance_different_lenght(sample1, sample2, wSlopeError, wCurvatureError, sisterCount)
    
    return distance


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
    x = series1-series2
    slope_e = np.square(np.around(np.absolute(x/10)))
    curv_e = np.square(np.minimum(np.remainder(x,10), 10-np.remainder(x,10)))
    error = wDim1*slope_e+wDim2*curv_e
    
    error = np.sum(error)
    return error/series1.shape[0]

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
    
    length1 = series1.shape[0]
    length2 = series2.shape[0]
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
#     error = np.square(sisters - longFV.T[np.newaxis,:,:])
#     weights = np.array([wDim1, wDim2])
#     
    slope_e = np.square(np.around(np.absolute(sisters-longFV.T[np.newaxis,:])/10))
    curv_e= np.square(np.minimum(np.remainder(sisters-longFV.T[np.newaxis,:],10), 10-np.remainder(sisters-longFV.T[np.newaxis,:],10)))
    error = wDim1*slope_e+wDim2*curv_e
    
#     error = error*weights[np.newaxis, np.newaxis, :]
#     
    # this is sort of stupid, it should be possible to do in one line
    # but axis=(1,2) does not work, it requires numpy 1.7. I have 1.6.1 
    # at the moment
    error = np.sum(error, axis=1)
    return np.min(error)/longFV.shape[0]


def create_sisters(shortFV, desired_shape, sister_count):
    '''
    Creates a set of new feature vectors that are behaviorally identical to the given 
    short feature vector (shortFV), and that have the stated number of segments (i.e. desired_shape).
    
    :param shortFV: The feature vector to be extended.
    :param desired_shape: The desired shape (2-by-number of sections) of the extended feature vectors (i.e. sisters) 
    :param sister_count: The desired number of sisters to be created
    ''' 
    
    #determine how much longer the vector has to become
    to_add = desired_shape[0]-shortFV.shape[0]
    
    #create a 2d array of indices
    indices = np.zeros(shape=(sister_count, desired_shape[0]),dtype=int)
    
    #fill the first part of the indices array with random numbers
    #these are the indices that will be used to extent the short vector
    indices[:, 0:to_add] = np.random.randint(0, 
                                             shortFV.shape[0], 
                                             size=(sister_count, to_add))
    
    #add the indices for the full vector to the rest
    indices[:, to_add::] = np.arange(0, shortFV.shape[0])[np.newaxis, :]
    
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


if __name__ == '__main__':
    print 'deneme'
    