'''
Created on Aug 21, 2013

@author: gonenc
'''
import numpy as np

def distance_dtw(sample1, sample2, wSlopeError=1, wCurvatureError=1):
    dtw = np.zeros([sample1.shape[0] + 1, sample2.shape[0] + 1])
    dtw[:, 0] = 1000
    dtw[0, :] = 1000
    dtw[0, 0] = 0
    for i in range(sample1.shape[0]):
        for j in range(sample2.shape[0]):
            cost = local_dist(sample1[i], sample2[j], wSlopeError, wCurvatureError)
            dtw[i + 1, j + 1] = cost + min([dtw[i + 1, j], dtw[i, j + 1], dtw[i, j]])
#             print i+1, j+1, dtw[i + 1, j + 1]
    max_len = max(sample1.shape[0], sample2.shape[0])
    return dtw[sample1.shape[0], sample2.shape[0]] / max_len

def local_dist(x, y, wDim1=1, wDim2=1):
    slope_e = np.square(x[0] - y[0])
    curv_e = np.square(x[1] - y[1])
    return  wDim1 * slope_e + wDim2 * curv_e

