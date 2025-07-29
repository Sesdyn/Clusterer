'''
Created on Aug 21, 2013
@author: gonenc

Updated July 25, 2025
'''
import numpy as np
from numba import njit

def distance_dtw(data_wo_labels):
    dRow = np.zeros(shape=(np.sum(np.arange(len(data_wo_labels))), ))

    runLogs = [({'Index': str(i)}, data_wo_labels[i]) for i in range(len(data_wo_labels))]

    dRow = compute_dtw_distances_numba(data_wo_labels, dRow)

    return dRow, runLogs


@njit
def compute_dtw_distances_numba(data_wo_labels, dRow):
    index = -1
    for i in range(len(data_wo_labels)):            
        for j in range(i+1, len(data_wo_labels)):
            index += 1
            distance = dtw_dist_numba(data_wo_labels[i], data_wo_labels[j]) 
            dRow[index] = distance
    return dRow


@njit
def dtw_dist_numba(sample1, sample2):
    """Numba-optimized DTW distance calculation"""
    dtw = np.zeros((sample1.shape[0] + 1, sample2.shape[0] + 1))
    dtw[:, 0] = np.inf #infinity is assigned instead of 1000
    dtw[0, :] = np.inf #infinity is assigned instead of 1000
    dtw[0, 0] = 0
    for i in range(sample1.shape[0]):
        for j in range(sample2.shape[0]):
            cost = np.absolute(sample1[i] - sample2[j])
            dtw[i + 1, j + 1] = cost + min(dtw[i + 1, j], dtw[i, j + 1], dtw[i, j])

    return dtw[sample1.shape[0], sample2.shape[0]]


if __name__ == '__main__':
    print('gonenc')