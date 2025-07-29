'''
Created on Dec 19, 2011
@author: gyucel

Updated July 25, 2025
'''
import numpy as np
from scipy.spatial.distance import pdist

def distance_triangle(data):
    '''
    The triangle distance is calculated as follows;
        Let ds1(.) and ds2(.) be two data series of length N. Then;
        A equals to the summation of ds1(i).ds2(i) from i=1 to N
        B equals to the square-root of the (summation ds1(i)^2 from i=1 to N)
        C equals to the square-root of the (summation ds2(i)^2 from i=1 to N)
        
        distance_triangle = A/(B.C)

     The triangle distance works only with data series of the same length
     
     In the literature, it is claimed that the triangle distance can deal with noise and amplitude scaling very well, and may yield poor
     results in cases of offset translation and linear drift.

    data: 2D array or np array; [[features1], [features2], ...]   ;;  Without labels
    return: 2D np array; [distance_vector, [({'Index': 0}, features1), ({'Index': 1}, features2), ...]]
    '''

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    cosine_distances = pdist(data, metric='cosine')
    dRow = 1 - cosine_distances

    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4,1),(2,2,6), (1.5,1,1)])
    #print(distance_triangle(tester))