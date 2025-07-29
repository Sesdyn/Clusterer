'''
Created on Dec 19, 2011
@author: gyucel

Updated July 25, 2025
'''
import numpy as np
from scipy.spatial.distance import pdist

def distance_mse(data):
    '''
    The MSE (mean squared-error) distance is equal to the SSE distance divided by the number of data points in data series.

    The SSE distance between two data series is equal to the sum of squared-errors between corresponding data points of these two data series.
    Let the data series be of length N; Then SSE distance between ds1 and ds2 equals to the sum of the square of error terms from 1 to N, 
    where error_term(i) equals to ds1(i)-ds2(i) 

    Given that SSE is calculated as given above, MSE equals SSE divided by N.

    As SSE distance, the MSE distance only works with data series of equal length.

    data: 2D array or np array; [[features1], [features2], ...]   ;;  Without labels
    return: 2D np array; [distance_vector, [({'Index': 0}, features1), ({'Index': 1}, features2), ...]]
    '''

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    sse_distances = pdist(data, metric='sqeuclidean')
    dRow = sse_distances / data.shape[1]

    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4,1),(2,2,6), (1.5,1,1)])
    #print(distance_mse(tester))