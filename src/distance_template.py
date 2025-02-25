'''
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

def distance_template(data_wo_labels):
    
    '''
    '''
    
    data_w_desc = []
    #Generates the feature vectors for all the time series that are contained in numpy array data
    dRow = np.zeros(shape=(np.sum(np.arange(len(data_wo_labels))), ))
    index = -1
    for i in range(data_wo_labels.shape[0]):
            
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
            distance = template_dist(data_wo_labels[i],data_wo_labels[j]) 
            dRow[index] = distance
    return dRow, data_w_desc

def template_dist(d1, d2):
    '''
    This is where you do your magic and specify how to calculate the distance
    '''
    return 0

if __name__ == '__main__':
    tester = np.array([(12,4),(2,2)])
