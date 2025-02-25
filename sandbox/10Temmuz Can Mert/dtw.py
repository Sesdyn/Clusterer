'''
Created on Aug 21, 2013

@author: gonenc
'''
import numpy as np

def distance_dtw(sample1, sample2, wSlopeError=1, wCurvatureError=1):
    dtw = np.zeros([sample1.shape[0] + 1, sample2.shape[0] + 1])
    dtw[:, 0] = 1000
    dtw[0, :] = 1000
    #dtw[:, 0] = np.inf
    #dtw[0, :] = np.inf
    dtw[0, 0] = 0
    #cost = local_dist_euclidian(sample1, sample2)
    for i in range(sample1.shape[0]):
        for j in range(sample2.shape[0]):
            #cost = local_dist(sample1[i], sample2[j], wSlopeError, wCurvatureError)
            cost = np.absolute(sample1[i] - sample2[j])
            dtw[i + 1, j + 1] = cost + min([dtw[i + 1, j], dtw[i, j + 1], dtw[i, j]])
            #print i+1, j+1, dtw[i + 1, j + 1]
    max_len = max(sample1.shape[0], sample2.shape[0])
    #return dtw[sample1.shape[0], sample2.shape[0]] / max_len
    return dtw[sample1.shape[0], sample2.shape[0]]

def local_dist_euclidian(x,y):
    diff = x-y
    s = np.sum(diff**2)
    res = np.sqrt(s)
    print res
    return res

def local_dist(x, y, wDim1=1, wDim2=1):
    dif = float(x - y)
    slope_e = np.square(np.around(np.absolute(dif / 10)))
    curv_e = np.square(np.minimum(np.remainder(dif, 10), 10 - np.remainder(dif, 10)))
    return  wDim1 * slope_e + wDim2 * curv_e

def read_tab(filename, ninputs):
    A = np.loadtxt(open(filename,"rb"),delimiter=",",skiprows=0, usecols=range(1, ninputs+1))
    return A


if __name__ == "__main__":
    filename = 'PatternSet_Basics.csv'
    ninputs = 101
    data = read_tab(filename, ninputs)
    x = data[0]
    y = data[1]
    z = data[2]
    print x
    print y
    print z
    print "{0:.10g}".format(distance_dtw(x, z))
    
