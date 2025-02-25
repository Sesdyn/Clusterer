'''
Created on Aug 21, 2013

@author: gonenc
'''
import numpy as np


def distance_dtw(data_wo_labels):
    data_w_desc = []
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
        #Feel free to add others to the behaviorDesc that can make your life easier after the clustering is done
        
        behavior = data_wo_labels[i]
        localLog = (behaviorDesc, behavior)
        data_w_desc.append(localLog)
    
        for j in range(i+1, len(data_wo_labels)):
            index += 1
            distance = dtw_dist(data_wo_labels[i],data_wo_labels[j]) 
            #print 'distance' + str(i) + ' vs ' + str(j)
            dRow[index] = distance
    return dRow, data_w_desc

def dtw_dist(sample1, sample2):
    
    dtw = np.zeros([sample1.shape[0] + 1, sample2.shape[0] + 1])
    dtw[:, 0] = np.inf #infinity is assigned instead of 1000
    dtw[0, :] = np.inf #infinity is assigned instead of 1000
    dtw[0, 0] = 0
    for i in range(sample1.shape[0]):
        for j in range(sample2.shape[0]):
            #cost = local_dist(sample1[i], sample2[j])
            cost = np.absolute(sample1[i] - sample2[j]) #absolute value of difference is used instead of local_dist function
            dtw[i + 1, j + 1] = cost + min([dtw[i + 1, j], dtw[i, j + 1], dtw[i, j]])
    max_len = max(sample1.shape[0], sample2.shape[0])
    return dtw[sample1.shape[0], sample2.shape[0]] #"/max_len" is deleted
def local_dist(x, y):
    dif = float(x - y)
    return  np.square(dif) 
