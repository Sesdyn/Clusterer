'''
Created on Dec 19, 2011
@author: gyucel

Updated July 25, 2025
'''
import numpy as np
from scipy.spatial.distance import pdist

def distance_sse(data):
    '''
    The SSE (sum of squared-errors) distance between two data series is equal to the sum of squared-errors between corresponding data points of these two data series.
    Let the data series be of length N; Then SSE distance between ds1 and ds2 equals to the sum of the square of error terms from 1 to N, 
    where error_term(i) equals to ds1(i)-ds2(i) 
    Since SSE calculation is based on pairwise comparison of individual data points, the data series should be of equal length.
    SSE distance equals to the square of Euclidian distance, which is a commonly used distance metric in time series comparisons.

    data: 2D array or np array; [[features1], [features2], ...]   ;;  Without labels
    return: 2D np array; [distance_vector, [({'Index': 0}, features1), ({'Index': 1}, features2), ...]]
    '''

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    dRow = pdist(data, metric='sqeuclidean')

    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4,1),(2,2,6), (1.5,1,1)])
    #print(distance_sse(tester))