'''
Created on Sep 6, 2013
Updated on Dec 10, 2016
@author: gyucel
'''

import logging as log
import time
from operator import mod
from numpy import gradient, polyfit
from dtw import *
log.basicConfig(filename='../output/Clusterer.log', format='%(levelname)s:%(message)s - %(asctime)s', datefmt='%H:%M:%S', filemode='w', level=log.DEBUG)
import numpy as np
from xlrd import open_workbook  # http://pypi.python.org/pypi/xlrd
#from dtw import distance_dtw
import matplotlib.pyplot as plt
#import clusterer as cl
import pandas as pd
#import behavior_splitter as bs

import pickle
#==============================================================================
# Imports raw data from sheet named "data" and their corresponding feature vectors from the sheet "features"
#==============================================================================
def import_data(inputFileName):
    relPathFileFolder = '../datasets'  # Relative path to the folder in which dataset files reside
    
    book = open_workbook(relPathFileFolder + '/' + inputFileName + '.xlsx')
    sheet_data = book.sheet_by_name('data')
    noSample = sheet_data.nrows - 1
    noFeat = sheet_data.ncols - 1
    print noSample, noFeat
    sheet_feat = book.sheet_by_name('features')
    
    # dataSet is a 4D list. Each entry is a 3D list object. 
    # First dimension is a string that keeps the label of the data series,  
    # Second dimension is a numpy array that keeps the actual data
    # Third dimension is a numpy array that keeps the feature vector
    dataSet = []
 
    for i in range(noSample):       
        row = sheet_data.row(i + 1)
        sample = []
        sample.append(row[0].value)  # gets the label of the data
        data = []
        for j in range(len(row) - 1):
            if row[j + 1].ctype == 0:
                break
            else:
                data.append(row[j + 1].value)
        sample.append(np.array(data))
        
        row2 = sheet_feat.row(2 * i + 1)
        row3 = sheet_feat.row(2 * i + 2)
        features = []
        features.append([])
        features.append([])
        for j in range(len(row2) - 1):
            if row2[j + 1].ctype == 0:
                break
            else:
                features[0].append(row2[j + 1].value)
                features[1].append(row3[j + 1].value)
        sample.append(np.array(features))
        dataSet.append(sample)
    return dataSet



'''
Calculates slope and curvature by fitting a second-order polynomial to 2% slices of the original data
'''
def sc_polyfit(dataSeries, medianFilter = False):
    dsLength = dataSeries.shape[0]
    
    cw = int(round (0.01 * dsLength, 0))
    if cw == 0:
        cw = 1
        
    sLength = dsLength - 2 * cw
    
    slope = np.zeros(sLength)
    curvature = np.zeros(sLength)
    
    
    for i in range(sLength):
        x = i + np.array(range(2 * cw + 1))
        y = dataSeries[x]
        # print i, x, y
        a = np.polyfit(x, y, 1)
        slope [i] = a[0]
        b = np.polyfit(x, y, 2)
        curvature [i] = b[0]
    
    if medianFilter:
        slope = filter_median(slope)
        curvature = filter_median(curvature)
        
    # Crop the raw array so that it alligns well with the slope array
    raw_data = dataSeries[cw:dsLength - cw:1]
    
    return raw_data, slope, curvature, 



def sc_gradient(dataSeries, medianFilter = False):
    
    slope = np.gradient(dataSeries)
    if medianFilter:
        slope = filter_median(slope)
    
    curvature = np.gradient(slope[1:-1])
    
    if medianFilter:
        curvature = filter_median(curvature)
    
    curvature = curvature [1:-1]
    
    slope_cut = (len(slope) - len(curvature)) /2
    data_cut = (len(dataSeries) - len(curvature)) /2
    
    slope = slope [slope_cut:-slope_cut]
    dataSeries = dataSeries[data_cut:-data_cut]
    
    return dataSeries, slope, curvature


 
def sc_difference(dataSeries, medianFilter = False):
    slope = np.zeros(len(dataSeries) - 1)
    for j in range(len(slope)):
        slope[j] = dataSeries[j + 1] - dataSeries[j]
    
    if medianFilter:
        slope = filter_median(slope)

    curvature = np.zeros(len(slope) - 1)
    for j in range(len(curvature)):
        curvature[j] = slope[j + 1] - slope[j]
    
    if medianFilter:
        curvature = filter_median(curvature)        

    dataCut = len(dataSeries) - len(curvature)
    dataSeries = dataSeries[dataCut::]
    slopeCut = len(slope) - len(curvature) 
    slope = slope[slopeCut::]
    
    return dataSeries, slope, curvature
    
'''
Conventional median filter with a window-width of fw
'''
def filter_median(raw_data, fw=5):
    rawLength = raw_data.shape[0]
    filteredLength = rawLength - 2 * fw
    
    filtered = np.zeros(filteredLength)
    # filtered[0:fw] = raw_data[0:fw]
    # filtered[rawLength - fw:rawLength] = raw_data[rawLength - fw:rawLength]
     
    # filtered[rawLength - fw + 1:rawLength - 1:1] = raw_data[rawLength - fw + 1:rawLength - 1:1]
     
    
    for i in range(filteredLength):
        x = i + np.array(range(2 * fw + 1))
        y = raw_data[x]
        # print i, x, y
        filtered[i] = np.median(y)
      
    return filtered

'''
Constructs the feature vector fv using the slope and curvature arrays
fv[0][i] : Sign of the slope for section i
fv[1][i] : Sign of the curvature for section i
fv[2][i] : Ending point of the section i
fv[3][i] : Length of the section i
'''
def construct_fv(slope, curvature):
    signSlope = slope.copy()
    signSlope[signSlope > 0] = 1
    signSlope[signSlope < 0] = -1
    signSlope[signSlope == 0] = 0
    
    signCurvature = curvature.copy()
    signCurvature[signCurvature > 0] = 1
    signCurvature[signCurvature < 0] = -1
    signCurvature[signCurvature == 0] = 0         
    
    section_scores = 10 * signSlope + signCurvature
    temp = section_scores[1::] - section_scores[0:section_scores.shape[0] - 1]
    transPoints = np.nonzero(temp)
    numberOfSections = len(transPoints[0]) + 1
    
    # Adding two new dimension the the feature vector; the starting point and the length of the section
    # Mainly for experimentation purposes, both or one of them can be removed later    
    featureVector = np.zeros(shape=(numberOfSections, 5))

    
    if numberOfSections == 1:
        featureVector[0][0] = signSlope[0]
        featureVector[0][1] = signCurvature[0]
        featureVector[0][2] = 0
        featureVector[0][3] = len(signSlope) - 1
    else:
        for k in range(len(transPoints[0])):
            featureVector[k][0] = signSlope[transPoints[0][k]]
            featureVector[k][1]= signCurvature[transPoints[0][k]]
            if k == 0:
                featureVector[k][2] = 0
            else:
                featureVector[k][2]= transPoints[0][k-1]
            featureVector[k][3]= transPoints[0][k]
            
        featureVector[numberOfSections - 1][0] = signSlope[-1]
        featureVector[numberOfSections - 1][1] = signCurvature[-1]
        featureVector[numberOfSections - 1][2] = transPoints[0][-1]
        featureVector[numberOfSections - 1][3] = len(signSlope) - 1
    
    for ind in range(numberOfSections):
        featureVector[ind][4] = featureVector[ind][3] - featureVector[ind][2]
        
    return featureVector



def filter_series(series, parentSeries, thold):
    '''
    Filters out a given time-series for insignificant fluctuations. For 
    example very small fluctuations due to numeric error of the simulator).
    '''
     
    absParent = np.absolute(parentSeries[0:parentSeries.shape[0] - 1])
    absSeries = np.absolute(series)
#     cond = absSeries < thold
    cond1a = np.not_equal(absParent, 0)
    cond1b = absSeries < thold * absParent
    cond2a = np.logical_not(cond1a)
    cond2b = absSeries < thold / 10
    cond1 = np.logical_and(cond1a, cond1b)
    cond2 = np.logical_and(cond2a, cond2b)
    cond = np.logical_or(cond1, cond2)
    series[cond] = 0
    return series

def extend_mids(vector):
    sections = vector[0].size
    added = 0
    for i in range(sections - 1):
        if(vector[0][i + added] * vector[0][i + 1 + added] == -1) and\
          (vector[0][i + added + 1] != 0):
            vector = np.insert(vector, i + 1 + added, 0, axis=1)
            vector[1][i + 1 + added] = vector[1][i + added]
            added += 1
    return vector

def extend_ends(vector):
    sections = vector[0].size
    added = 0
    
    # Front extension   
    if(vector[0][0] == 0 and (vector[1][0] == 1 or vector[1][0] == -1)):
        vector = np.insert(vector, 0, 0, axis=1)
        added += 1
    # End extension
    if((vector[0][sections - 2 + added] * vector[1][sections - 2 + added] == -1) and (vector[0][sections - 1 + added] == 0)):
        vector = np.append(vector, [[0], [0]], axis=1)
        added += 1
    return vector



def analyze(data):
    prior_round = 0
    post_round = 0
    
    ts = data[1]
    
    if prior_round:
        ts = np.round(ts,prior_round)
                
    ts, slope, curvature = sc_difference(ts, True)
            
    if post_round > 0:
        slope = np.round(slope,post_round)
        curvature = np.round(curvature,post_round)       
            
    feature = construct_fv(slope, curvature)
    
#     print data[0]
#     for k in feature[feature[:,4]>1]:
#         print k  
    return feature  

def compare_fv(fv,real_fv):
    
    if np.array_equal(fv, real_fv):
        return 0
    elif fv.shape[0] <= (real_fv.shape[0]+2):
        return 0.5
    else:
        return 1 
    
def generate_real_fv(dt):
    if dt == 'LNRGR':
        return np.array([[1,0]])
    elif dt == 'LNRDC':
        return np.array([[-1,0]])
    elif dt == 'PEXGR':
        return np.asanyarray([[1,1]])
    elif dt == 'NEXGR':
        return np.asanyarray([[1,-1]])
    elif dt == 'PEXDC':
        return np.asanyarray([[-1,-1]])
    elif dt == 'NEXDC':
        return np.asanyarray([[-1,1]])
    elif dt == 'SSHGR':
        return np.asanyarray([[1,1],[1,-1]])
    elif dt == 'SSHDC':
        return np.asanyarray([[-1,-1],[-1,1]])
    elif dt == 'GR1D1':
        return np.asanyarray([[1,-1],[0,0],[-1,-1]])
    elif dt == 'GR1D2':
        return np.asanyarray([[1,-1],[-1,-1],[-1,1]])
    elif dt == 'GR2D1':
        return np.asanyarray([[1,1],[1,-1],[-1,-1]])
    elif dt == 'GR2D2':
        return np.asanyarray([[1,1],[1,-1],[-1,-1],[-1,1]])
        
if __name__ == '__main__':
    
#     data = cl.import_all_files()
#     data.to_pickle('C:\Users\sesdyn12\git\clusterer\datasets\\all_data.pkl')
#     print "ok"
#     raw_input()

#reload object from file

  #=============================================================================
  #   data = pd.read_pickle('../datasets/all_data.pkl')
  #     
  # 
  #   data_values = data.iloc[:,1:len(data.columns)-3].as_matrix()
  #   data_id = data.iloc[:,0].as_matrix()
  #   data_dataset = data.iloc[:,len(data.columns)-1].as_matrix()
  #   data_type = data.iloc[:,len(data.columns)-2].as_matrix()
  # 
  #   dataset_errors=[]
  #   # samples[index of the sample][0 - label; 1 - data ; 2 - ideal feature vector]
  # 
  #   data['Error']=np.nan
  #   for i in range(data.shape[0]):
  #       print i
  #       c_data=[]
  #       c_data.append('lgr-001')
  #       c_data.append(data_values[i,])
  #       fv = analyze(c_data) 
  #       fv = fv[:,range(2)]
  #       real_fv = generate_real_fv(data_type[i])  
  #       #res = compare_fv(fv,real_fv)
  #       res2 = distance_dtw(fv,real_fv)
  #       data['Error'].iloc[i]=res2
  #   print data   
  #   data.to_pickle('../datasets/all_data_error_dtw.pkl')
  #=============================================================================
 
    data = pd.read_pickle('../datasets/all_data_error.pkl') #Referansi relative'e cevirdim, absolute path yerine
    print data.groupby(['Source', 'Class'], as_index=False)['Error'].mean()
    print data.groupby(['Source', 'Class'], as_index=False)['Error'].max()
    print data.groupby(['Source', 'Class'], as_index=False)['Error'].min()


                     