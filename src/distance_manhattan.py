'''
Created on Sep 9, 2014
@author: cansucullu

Updated July 25, 2025
'''
import numpy as np
from scipy.spatial.distance import pdist

def distance_manhattan(data_wo_labels):
    '''
    The Manhattan distance between two data series is equal to the sum of absolute differences between corresponding data points of these two data series.
    Let the data series be of length N; Then Manhattan distance between ds1 and ds2 equals to the sum of the absolute values of error terms from 1 to N, 
    where error_term(i) equals to ds1(i)-ds2(i) 
    The Manhattan distance only works with data series of equal length. It is also referred as rectilinear distance, L1 distance or city block distance

    data: 2D array or np array; [[features1], [features2], ...]   ;;  Without labels
    return: 2D np array; [distance_vector, [({'Index': 0}, features1), ({'Index': 1}, features2), ...]]
    '''

    if not isinstance(data_wo_labels, np.ndarray):
        data_wo_labels = np.array(data_wo_labels)

    runLogs = [({'Index': str(i)}, data_wo_labels[i]) for i in range(len(data_wo_labels))]

    dRow = pdist(data_wo_labels, metric='cityblock')
    
    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4,1),(2,2,6), (1.5,1,1)])
    result = distance_manhattan(tester)
    print(result[0])  # Print the distance row
    for log in result[1]:
        print(log[0])  # Print the description of each run
        print(log[1])  # Print the data series for each run

