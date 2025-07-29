'''
Created on Dec 19, 2011
Revised on July 29, 2025
Revised by: gyucel
This module implements the MSE (mean squared-error) distance between two data series.
The MSE distance is equal to the SSE (sum of squared-errors) distance divided by the number of data points in the data series.
The MSE distance only works with data series of equal length.
'''

import numpy as np


def msedist(d1,d2):
    sse = ((d1-d2)**2).sum()
    mse = np.average(sse)
    return mse

def distance_mse(data):
    '''
    The MSE (mean squared-error) distance is equal to the SSE distance divided by the number of data points in data series.
    
    The SSE distance between two data series is equal to the sum of squared-errors between corresponding data points of these two data series.
    Let the data series be of length N; Then SSE distance between ds1 and ds2 equals to the sum of the square of error terms from 1 to N, 
    where error_term(i) equals to ds1(i)-ds2(i) 
    
    Given that SSE is calculated as given above, MSE equals SSE divided by N.
    
    As SSE distance, the MSE distance only works with data series of equal length.
    '''
    
    runLogs = []
    #Generates the feature vectors for all the time series that are contained in numpy array data
    dRow = np.zeros(shape=(np.sum(np.arange(len(data))), ))
    index = -1
    for i in range(len(data)):
            
        # For each run, a log is created
        # Log includes a description dictionary that has key information 
        # for post-clustering analysis, and the data series itself. These 
        # logs are stored in a global array named runLogs
        behaviorDesc = {}
        behaviorDesc['Index'] = str(i)
        
        behavior = data[i]
        localLog = (behaviorDesc, behavior)
        runLogs.append(localLog)
    
        for j in range(i+1, len(data)):
            index += 1
            distance = msedist(data[i],data[j]) 
            dRow[index] = distance
    return dRow, runLogs

if __name__ == '__main__':
    tester = np.array([(12,4,2),(2,2,1),(1,1,1)])
    result = distance_mse(tester)
    print(result[0])  # Print the distance row
    for log in result[1]:  # Print the logs
        print(log[0])
        print(log[1])
