'''
Created on Aug 19, 2013

@author: gyucel
'''

import logging as log
import time
log.basicConfig(filename='../output/Clusterer.log', format='%(levelname)s:%(message)s - %(asctime)s', datefmt='%H:%M:%S', filemode='w', level=log.DEBUG)

import numpy as np
from xlrd import open_workbook  # http://pypi.python.org/pypi/xlrd
from pattern import distance_pattern
from dtw import distance_dtw


def import_data(inputFileName):
    relPathFileFolder = '../datasets'  # Relative path to the folder in which dataset files reside
    
    book = open_workbook(relPathFileFolder + '/' + inputFileName + '.xlsx')
    sheet = book.sheet_by_name('data')
    noSample = sheet.nrows - 1
    noFeat = sheet.ncols - 1
    print noSample, noFeat
    
    
    # dataSet is a 3D list. Each entry is a 2D list object. First dimension is a string that keeps the label of the data series, and the second dimension is a numpy array that keeps the actual data
    dataSet = []
 
    for i in range(noSample):       
        row = sheet.row(i + 1)
        sample = []
        sample.append(row[0].value)  # gets the label of the feature
        data = []
        for j in range(len(row) - 1):
            if row[j + 1].ctype == 0:
                break
            else:
                data.append(row[j + 1].value)
        sample.append(np.array(data))
        dataSet.append(sample)
    
    return dataSet




def populate_dRow_dtw(samples, wSlopeError=1, wCurvatureError=1):
    dRow_dtw = np.zeros(shape=(np.sum(np.arange(len(samples))),))
    index = -1
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            index += 1
            dRow_dtw[index] = distance_dtw(samples[i][1], samples[j][1], wSlopeError, wCurvatureError)
    return dRow_dtw

def populate_dRow_pattern(samples, sSlopeError, wCurvatureError):
    dRow_pattern = np.zeros(shape=(np.sum(np.arange(len(samples))),))
    index = -1
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            index += 1
            dRow_pattern[index] = distance_pattern(samples[i][1], samples[j][1], wSlopeError, wCurvatureError)
    return dRow_pattern

if __name__ == '__main__':
    samples = import_data("SampleFeatures")
       
    wSlopeError = 1
    wCurvatureError = 1
#     start_time = time.time()
    dRow_pattern = populate_dRow_pattern(samples, wSlopeError, wCurvatureError)
#     print time.time() - start_time, "seconds"
#     start_time = time.time()
    dRow_dtw = populate_dRow_dtw(samples, wSlopeError, wCurvatureError)
#     print time.time() - start_time, "seconds"

#     import timeit
#     print(timeit.timeit("dRow_pattern = populate_dRow_pattern(samples, wSlopeError, wCurvatureError)", setup="from __main__ import populate_dRow_pattern"))
#     print(timeit.timeit("dRow_dtw = populate_dRow_dtw(samples, wSlopeError, wCurvatureError)", setup="from __main__ import populate_dRow_dtw"))
#     
    difference = dRow_pattern - dRow_dtw
    print np.min(difference), np.max(difference), np.average(difference)
    print difference
    
