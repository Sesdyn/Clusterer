'''
Updated July 25, 2025

Docstring comes here

Requirements for a proper new distance module;
    * The module is required to have method that obeys the standard distance naming convention (distance_[name]). For example, if the distance to be defined is MSE, a good name for the method would be distance_mse
    * The aforementioned method should take the raw dataset as the input. 
    * It has to return two things: dRow and data_w_desc
    * dRow: Distance row, that corresponds to the upper triangle of the pairwise distances matrix
    * data_w_desc: A list that contains the original data, as well as a descriptor dictionary for each dataseries in the original set
    * Descriptor dictionary contains all information about the dataseries with respect to the distance being considered. Only required element is the 'Index' which is the original index of the dataseries
'''
import numpy as np
from scipy.spatial.distance import pdist

def distance_template(data):
    '''
    data: 2D array or np array; [[features1], [features2], ...]   ;;  Without labels
    return: 2D np array; [distance_vector, [({'Index': 0}, features1), ({'Index': 1}, features2), ...]]
    '''

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    runLogs = [({'Index': str(i)}, data[i]) for i in range(len(data))]

    # Implement the distance calculation here. Make it return a condensed distance matrix, which is a 1D array 
    # containing the upper triangular portion (excluding diagonal) of the full pairwise distance matrix.
    dRow = 0

    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4,1),(2,2,6), (1.5,1,1)])
    #print(distance_sse(tester))