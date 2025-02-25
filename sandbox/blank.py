'''
Created on Sep 23, 2013

@author: gyucel
'''

from dtw import distance_dtw
import numpy as np

if __name__ == '__main__':
    sample1 = np.array([9,11,9,11,9,11,9,11,9,11,9,11,9,-1])
    sample2 = np.array([9,0])
    d = distance_dtw(sample1, sample2)
    print d
    