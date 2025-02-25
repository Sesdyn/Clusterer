'''
Created on Sep 9, 2014

@author: cansucullu
'''
import numpy as np

def manhattandist(d1, d2):
    d = np.abs(d1-d2).sum()
    return d

def distance_manhattan(data_wo_labels):
    '''
    The Manhattan distance between two data series is equal to the sum of absolute differences between corresponding data points of these two data series.
    Let the data series be of length N; Then Manhattan distance between ds1 and ds2 equals to the sum of the absolute values of error terms from 1 to N, 
    where error_term(i) equals to ds1(i)-ds2(i) 
    
    The Manhattan distance only works with data series of equal length. It is also referred as rectilinear distance, L1 distance or city block distance
    '''
    
    runLogs = []
    #Generates the feature vectors for all the time series that are contained in numpy array data
    dRow = np.zeros(shape=(np.sum(np.arange(len(data_wo_labels))), ))
    index = -1
    for i in range(len(data_wo_labels)):
            
        # For each run, a log is created
        # Log includes a description dictionary that has key information 
        # for post-clustering analysis, and the data series itself. These 
        # logs are stored in a global array named runLogs
        behaviorDesc = {}
        behaviorDesc['Index'] = str(i)
        
        behavior = data_wo_labels[i]
        localLog = (behaviorDesc, behavior)
        runLogs.append(localLog)
    
        for j in range(i+1, len(data_wo_labels)):
            index += 1
            distance = manhattandist(data_wo_labels[i],data_wo_labels[j]) 
            dRow[index] = distance
    return dRow, runLogs


if __name__ == '__main__':
    tester = np.array([(12,4,1),(2,2,6)])
