'''
Created on Dec 16, 2014
Updated 23
@author: gonenc
'''
import numpy as np
from sklearn import datasets, linear_model

def construct_features(data, significanceLevel=0.01):
    '''
    Constructs a feature vector for each of the data-series contained in the 
    data. 
    
    '''
    # TODO, the casting of each feature to a list of tuples might be 
    # removed at some stage, it will lead to a speed up, for you 
    # can vectorize the calculations that use the feature vector
    features = []
    for i in range(len(data)):
        feature = split_behavior(data[i])
        features.append(feature)
    return features


def split_behavior(dataSeries, significanceLevel=0.01):
    '''
    Splits the given dataSeries into sections of different atomic behavior modes, and returns the array of these sections. Each element in 
    this 2-D array represents a section along the time-series that can be characterized as an atomic behaviour mode.
    '''
    
#     regr = linear_model.LinearRegression()
# 
#     # Train the model using the training sets
#     regr.fit(np.asanyarray(range(dataSeries.shape[0])).reshape(-1,1), np.asanyarray(dataSeries).reshape(-1,1))
#     predicted_series = regr.predict(np.asanyarray(range(dataSeries.shape[0])).reshape(-1,1))
#     
#     np.transpose(predicted_series).tolist()[0]
    
    slope = np.gradient(dataSeries)
    curvature = np.gradient(slope)
    
    curvature[np.absolute(curvature)<np.absolute(slope)*significanceLevel]=0
    slope[np.absolute(slope)<np.absolute(dataSeries)*significanceLevel]=0
   
                    
    signSlope = slope.copy()
    signSlope[signSlope>0] = 1
    signSlope[signSlope<0] = -1
    signSlope[signSlope==0] = 0
    
    signCurvature = curvature.copy()
    signCurvature[signCurvature>0] = 1
    signCurvature[signCurvature<0] = -1
    signCurvature[signCurvature==0] = 0         
    
    sections = 10*signSlope+signCurvature
    raw_input()
    temp = sections[1::]-sections[0:sections.shape[0]-1]
    transPoints = np.nonzero(temp)
    numberOfSections = len(transPoints[0])+1
    
    featureVector = np.zeros(shape=(2,numberOfSections))
    
    for k in transPoints:
        featureVector[0][0:len(featureVector[0])-1] = signSlope[k]
        featureVector[1][0:len(featureVector[0])-1] = signCurvature[k]
    featureVector[0][numberOfSections-1]= signSlope[-1]
    featureVector[1][numberOfSections-1]= signCurvature[-1]
 
    # featureVector=extend_mids(featureVector)
    # featureVector=extend_ends(featureVector)
    
    return featureVector




def filter_series(series, parentSeries, thold):
    '''
    Filters out a given time-series for insignificant fluctuations. For 
    example very small fluctuations due to numeric error of the simulator).
    
    Not used anymore
    '''
     
    absParent = np.absolute(parentSeries[0:parentSeries.shape[0]-1])
    absSeries = np.absolute(series)
    cond = absSeries < thold
    cond1a = np.not_equal(absParent, 0)
    cond1b = absSeries < thold*absParent
    cond2a = np.logical_not(cond1b)
    cond2b = absSeries < thold/10
    cond1 = np.logical_and(cond1a,cond1b)
    cond2 = np.logical_and(cond2a,cond2b)
    #cond = np.logical_or(cond1,cond2)
    series[cond] = 0
    return series


def extend_mids(vector):
    '''
    Not used anymore 15/12/14
    '''
    sections = vector[0].size
    added = 0
    for i in range(sections-1):
        if(vector[0][i+added]*vector[0][i+1+added]==-1) and\
          (vector[0][i+added+1]!=0):
            vector = np.insert(vector, i+1+added, 0, axis=1)
            vector[1][i+1+added] = vector[1][i+added]
            added+=1
    return vector

def extend_ends(vector):
    '''
    Not used anymore 15/12/14
    '''
    sections = vector[0].size
    added = 0
    
    if(vector[0][0]*vector[1][0]==1):
        #Front extension
        vector = np.insert(vector, 0, 0, axis=1)
        vector[1][0] = vector[1][1]
        added+=1
    if(vector[0][sections-1+added]*vector[1][sections-1+added]==-1):
        #End extension
        vector = np.append(vector, [[0],[0]], axis=1)
        vector[1][sections+added] = vector[1][sections-1+added]
        added+=1
    return vector




'''
The main method where the user needs to specify the path to the simulation results
'''   
if __name__ == '__main__':
    print 'gonenc'