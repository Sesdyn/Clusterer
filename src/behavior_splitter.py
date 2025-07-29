'''
Created on Dec 16, 2014
Updated 23
@author: gonenc

Updated July 25, 2025
'''
import numpy as np

def construct_features(data, significanceLevel=0.01):
    '''
    Constructs a feature vector for each of the data-series contained in the data.

    data: 2D array or np array; [[features1], [features2], ...]
    return: 3D np array; [[[signSlopeFeature1Section1, signCurvatureFeature1Section1], [signSlopeFeature1Section2, signCurvatureFeature1Section2], ...], ...]
    '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    features = []

    slopes = np.gradient(data, axis=1)
    curvatures = np.gradient(slopes, axis=1)

    for i in range(data.shape[0]):
        feature = split_behavior(data[i], slopes[i], curvatures[i], significanceLevel)
        features.append(feature)

    return features

def split_behavior(dataSeries, slope, curvature, significanceLevel=0.01):
    '''
    Splits the given dataSeries into sections of different atomic behavior modes. Each element in 
    this 2-D array represents a section along the time-series that can be characterized as an atomic behaviour mode.

    dataSeries: 1D np array; features1
    slope: 1D np array; slope1 (slope of features1)
    curvature: 1D np array; curvature1 (curvature of features1)

    return: 2D np array; [[signSlopeSection1, signCurvatureSection1], [signSlopeSection2, signCurvatureSection2], ...
    '''

    abs_data = np.abs(dataSeries)
    abs_slope = np.abs(slope)
    abs_curvature = np.abs(curvature)
    
    data_threshold = abs_data * significanceLevel
    slope_threshold = abs_slope * significanceLevel

    slope = slope * (abs_slope >= data_threshold)
    curvature = curvature * (abs_curvature >= slope_threshold)

    signSlope = np.sign(slope)
    signCurvature = np.sign(curvature)

    sections = signSlope * 10 + signCurvature

    transitions = np.diff(sections)
    transPoints = np.flatnonzero(transitions)
    numberOfSections = len(transPoints) + 1

    featureVector = np.empty((2, numberOfSections))

    if numberOfSections == 1:
        featureVector[0, 0] = signSlope[0]
        featureVector[1, 0] = signCurvature[0]
    else:
        section_starts = np.concatenate(([0], transPoints + 1))
        featureVector[0] = signSlope[section_starts]
        featureVector[1] = signCurvature[section_starts]

    return featureVector

'''
The main method where the user needs to specify the path to the simulation results
'''
if __name__ == '__main__':
    print('gonenc')