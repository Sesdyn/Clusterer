'''
Pattern-oriented behavior clustering module.
The module provides the methods for importing, pre-processing, clustering and post-processing of bundles of time-series data (e.g. from a sensitivity analysis) 
    * Created on June 28, 2013
    * Major revision on August 13, 2013 - TU Delft EMA Workbench dependencies are removed
    * Major revision on August 09, 2014 - Enthought dependencies are removed
.. codeauthor:: 
     gyucel <gonenc.yucel (at) boun (dot) edu (dot) tr>,

Updated July 30, 2025
'''
from __future__ import division
import logging as log
#log.basicConfig(filename='../output/Clusterer.log', format='%(levelname)s:%(message)s - %(asctime)s', datefmt='%H:%M:%S', filemode='w', level=log.DEBUG)

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from xlrd import open_workbook, XLRDError #http://pypi.python.org/pypi/xlrd
import xlsxwriter
import os
import time
import pandas as pd
from distance_pattern import distance_pattern
from distance_pattern_dtw import distance_pattern_dtw
from distance_mse import distance_mse
from distance_sse import distance_sse
from distance_dtw import distance_dtw
from distance_manhattan import distance_manhattan
from behavior_splitter import construct_features

varName = ""
clusterCount = 0

# The cluster method only recognizes the distances that are listed in the distance_functions dictionary
distance_functions = {'pattern': distance_pattern, 'pattern_dtw': distance_pattern_dtw,'mse': distance_mse, 'sse': distance_sse, 'dtw': distance_dtw, 'manhattan': distance_manhattan}


def import_data(inputFileName, withClusters = False):
    '''
    Method that imports dataseries to be analyzed from .xlsx files. Unless specified otherwise, looks for the file in the datasets folder of the project. Optionally it can also read the original clusters of the dataseries. For that, the input file should contain a sheet names *clusters*, and the order of the dataseries in this sheet should be identical to the sorting in the data sheet

    :param inputFileName: The name of the .xlsx file that contains the dataset
    :param withClusters: If True, checks the sheet names *clusters* and returns also the original clusters/classess of dataseries
    :returns: Two lists. The first one contains 2D lists, each corresponding to a single dataseries. The first entry is a string that keeps the label of the sample, and the second entry is a numpy array that keeps the data. The second list is optional, and returns when *withClusters* is True. It contains the original clusters of the input data  
    :rtype: List (3D)
    '''
    
    relPathFileFolder = '../datasets'    #Relative path to the folder in which dataset files reside
    book = open_workbook(relPathFileFolder+'/'+inputFileName+'.xlsx')
    sheet_data = book.sheet_by_name('data')
    noRuns = sheet_data.nrows-1
    
    #dataSet is a 3D list. Each entry is a 2D list object. First dimension is a string that keeps the label of the data series, and the second dimension is a numpy array that keeps the actual data
    all_rows = [sheet_data.row_values(i+1) for i in range(noRuns)]
    
    # Vectorized separation of labels and data
    labels = [row[0] for row in all_rows]
    data_arrays = [np.array(row[1:]) for row in all_rows]
    
    data_w_desc = list(zip(labels, data_arrays))
    
    if withClusters:
        try:
            sheet_clusters = book.sheet_by_name('clusters')
            clusters_original = [sheet_clusters.cell(i+1, 1).value for i in range(noRuns)]
        except XLRDError:
            clusters_original = ['NA'] * noRuns
        return data_w_desc, clusters_original 
    else:
        return data_w_desc


def import_all_files():
    names = ["StellaFreeFloat","StellaDefault","Vensim"]
    dataframes = [import_all(n) for n in names]
    full_data = pd.concat(dataframes, ignore_index=True)
    return full_data


def import_all(name):
    relPathFileFolder = '../datasets'
    inputFileName='Dataset_Basic_'+name
    book = open_workbook(relPathFileFolder+'/'+inputFileName+'.xlsx')
    names = book.sheet_names()
    dataframes = [import_pandas_data(name, n) for n in names]
    complete_data = pd.concat(dataframes, ignore_index=True)
    complete_data['Source'] = name
    return complete_data


def import_pandas_data(name, cls):
    relPathFileFolder = '../datasets'
    inputFileName='Dataset_Basic_'+name
    book = open_workbook(relPathFileFolder+'/'+inputFileName+'.xlsx')
    sheet_data=book.sheet_by_name(cls)
    noRuns = sheet_data.nrows-1
    
    data = [sheet_data.row_values(i+1) for i in range(noRuns)]
    data = pd.DataFrame(np.array(data))
    data = data.rename(columns = {0:'Id'})
    data["Class"] = cls
    
    return data


def normalize_data(data_w_labels):
    '''
    Compute the normalized version of the time-series data such that
    y_i = x_i - min(x) / (max(x) - min(x))
    :param data: 1-D or 2-D numpy ndarray, where the first column has description information
    :returns: ndarray, Normalized input data
    '''
    # Extract all data arrays, normalize them in batch, then reassign
    data_arrays = np.array([item[1] for item in data_w_labels])
    
    mins = np.min(data_arrays, axis=1, keepdims=True)
    maxs = np.max(data_arrays, axis=1, keepdims=True)
    normalized_arrays = (data_arrays - mins) / (maxs - mins)
    
    result = [[item[0], normalized_arrays[i]] for i, item in enumerate(data_w_labels)]
    
    return result


def standardize_data(data_w_labels):
    '''
    Compute the standardized version of the time-series data such that
    y_i = x_i - mean(x) / std(x)
    :param data: 1-D or 2-D numpy ndarray, where the first column has description information
    :returns: ndarray, Standardized input data
    '''
    # Extract all data arrays, standardize them in batch, then reassign
    data_arrays = np.array([item[1] for item in data_w_labels])

    means = np.mean(data_arrays, axis=1, keepdims=True)
    stds = np.std(data_arrays, axis=1, keepdims=True)
    standardized_arrays = (data_arrays - means) / stds

    result = [[item[0], standardized_arrays[i]] for i, item in enumerate(data_w_labels)]

    return result


def cluster(data_w_labels,  distance='pattern', interClusterDistance='complete',
            cMethod='inconsistent', cValue=1.5, plotDendrogram=False, **kwargs):
    '''
    Method that clusters time-series data based on the specified distance measure using a hierarchical clustering algorithm. Optionally the method also plots the dendrogram generated by the clustering algorithm
    
    :param data: A list of lists. Each entry of the master list corresponds to a dataseries. The second order lists have two entries: The first entry is the label of the dataseries, and the second entry is a numpy array that keeps the data
    :param str distance: The distance metric to be used. Default value is *'pattern'*
    :param str interClusterDistance: How to calculate inter cluster distance.
                                 see `linkage <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage>`_ 
                                 for details. Default value is *'inconsistent'*
    :param cMethod: Cutoff method, 
                    see `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
                    for details.
    :param cValue: Cutoff value, see 
                   `fcluster <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster>`_ 
                   for details.
    :param plotDendogram: Boolean, if true, plot dendogram.
    :returns: A tuple containing the list of distances (i.e. dRow), the list of Cluster objects (i.e. clusterList : for each cluster a Cluster object that contains basic info about the cluster), 
            and a list that gives the index of the cluster each data series is allocated (i.e. clusters).     
    :rtype: Tuple
    
    The remainder of the arguments are passed on to the specified distance 
    function.
    
    Pattern Distance:
    
    * 'distance': String that specifies the distance to be used. 
                  Options: pattern (default), mse, sse, triangle
    * 'filter?': Boolean that specifies whether the data series will be 
                 filtered (for bmd distance)
    * 'slope filter': A float number that specifies the filtering threshold 
                     for the slope (for every data point if change__in_the_
                     outcome/average_value_of_the_outcome < threshold, 
                     consider slope = 0) (for bmd distance)
    * 'curvature filter': A float number that specifies the filtering 
                          threshold for the curvature (for every data point if 
                          change__in_the_slope/average_value_of_the_slope < 
                          threshold, consider curvature = 0) (for bmd distance)
    * 'no of sisters': 50 (for pattern distance)

    '''
    # Construct a list that includes only the data part. Gets rid of the label string in dataSet[i][0]
    data_wo_labels = [item[1] for item in data_w_labels]
    
    # Construct a list with distances. This list is the upper triangle
    # of the distance matrix
    dRow, data_w_desc = construct_distances(data_wo_labels, distance, **kwargs)

    # Allocate individual runs into clusters using hierarchical agglomerative 
    # clustering. clusterSetup is a dictionary that customizes the clustering 
    # algorithm to be used.
    
    clusters, data_w_desc = flatcluster(dRow, data_w_desc, plotDendrogram=plotDendrogram, 
                                        interClusterDistance=interClusterDistance, cMethod=cMethod, cValue=cValue)

    clusterList = create_cluster_list(clusters, dRow, data_w_desc)
    return dRow, clusterList, clusters


def create_cluster_list(clusters, distRow, data_w_desc):
    '''  
    Given the results of a clustering, the method creates Cluster objects for each of the identified clusters. Each cluster object contains member data series, as well as a sample/representative dataseries
    
    :param clusters: A list that contains the cluster number of the corresponding dataseries in the dataset(If the clusters[5] is 12, data[5] belongs to cluster 12 
    :param distRow: The row of distances coming from the distance function
    :param data_w_desc: The list that contains the raw data as well as the descriptor dictionary for each data series 
    
    :returns: A list of Cluster objects
    :rtype: List
    '''
    
    nr_clusters = np.max(clusters)
    cluster_list = []
    total_size = clusters.shape[0]
    
    for i in range(1, nr_clusters+1):
        # Determine the indices for cluster i
        indices = np.where(clusters==i)[0]
        cluster_size = indices.shape[0]
        
        if cluster_size == 1:
            originalIndex = indices[0]
            cluster = Cluster(i,  indices, data_w_desc[originalIndex], [data_w_desc[originalIndex]])
            cluster_list.append(cluster)
            continue
        
        q_flat, r_flat = np.triu_indices(cluster_size, k=1)
        
        indices_q = indices[q_flat]
        indices_r = indices[r_flat]
        
        i_vals = indices_r
        j_vals = indices_q
        drow_indices = total_size * j_vals - j_vals * (j_vals + 1) // 2 + i_vals - j_vals - 1
        
        #make a distance matrix
        dist_clust = distRow[drow_indices]
        dist_matrix = squareform(dist_clust)

        #sum across the rows
        row_sum = dist_matrix.sum(axis=0)
        
        #get the index of the result with the lowest sum of distances
        min_cIndex = row_sum.argmin()
    
        # convert this cluster specific index back to the overall cluster list 
        # of indices
        originalIndex = indices[min_cIndex]

        indices_list = indices.astype(int).tolist()
        
        cluster = Cluster(i, indices, data_w_desc[originalIndex], [data_w_desc[idx] for idx in indices_list])
        cluster_list.append(cluster)
        
    return cluster_list


def construct_distances(data_wo_labels, distance='pattern', **kwargs):
    """ 
    Constructs a row vector of distances (a condensed version of a n-by-n matrix of distances) for n data-series in data 
    according to the specified distance.
    
    Distance argument specifies the distance measure to be used. Options are as follows;
        * pattern: a distance based on qualitative dynamic pattern features 
        * sse: regular sum of squared errors
        * mse: regular mean squared error
        * triangle: triangular distance 
        * dtw: Dynamic time warping distance
    
    :param data: The list of dataseries to be clustered. Each entry is a numpy array that stores the data for a timeseries. 
    :param distance: The distance type to be used in calculating the pairwise distances. Default is *'pattern'* 
    :returns: A row vector of distances, and a list that stores the original data with distance-relevant dataseries descriptor 
    :rtype: Tuple (2 lists)
    """
    # Sets up the distance function according to user specification
    try:
        return distance_functions[distance](data_wo_labels, **kwargs)
    except KeyError:
        log.error('Unknown distance is used')
        print ('Unknown distance is used')
        raise

        
def flatcluster(dRow, data, interClusterDistance='complete', plotDendrogram=True, cMethod='inconsistent', cValue=2.5):
    z = linkage(dRow, interClusterDistance)
    
    if plotDendrogram:
        plotdendrogram(z)
    
    clusters = fcluster(z, cValue, cMethod)
    
    noClusters = max(clusters)
    
    #unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    #print('Total number of clusters:', noClusters)
    #for cluster_id, count in zip(unique_clusters, cluster_counts):
    #    print("Cluster", cluster_id, ":", count)
    
    global clusterCount
    clusterCount = noClusters
    
    cluster_strings = [str(cluster_id) for cluster_id in clusters]
    for i, log in enumerate(data):
        log[0]['Cluster'] = cluster_strings[i]
    
    return clusters, data


def plotdendrogram(z):
    dendrogram(z, truncate_mode='lastp', show_leaf_counts=True, show_contracted=True)
    plt.show()


def plot_clusters(cluster_list, dist, mode='show',fname='results'):
    '''
    Takes a list of Cluster objects as an input. Plots the members of each cluster on a seperate plot
    
    :param clusterList: deneme
    :param dataset:
    :param groupPlot:
    :param mode: default is show, use save to save the figure in png format
    :param fname: if mode=save option is used this is used as the filename, type without extension
    :rtype: Matplotlib graph
    '''
    main_fig = plt.figure(figsize=(14,10))
    #main_fig.suptitle('deneme')
    main_fig.canvas.manager.set_window_title(dist + ' distance')
    no_plots = len(cluster_list)
    no_cols = 4
    no_rows = int(np.math.ceil(float(no_plots) / no_cols))
    i = 1
    
    for clust in cluster_list:
        sub_plot = main_fig.add_subplot(no_rows, no_cols, i)
        i = i + 1
               
        #=======================================================================
        # # For plotting only the sample of each cluster 
        # t = np.array(range(clust.sample[1].shape[0]))
        # sub_plot.plot(t, clust.sample[1], linewidth=2)
        #=======================================================================
         
        for j in clust.members:
            t = np.array(range(j[1].shape[0]))
            sub_plot.plot(t, j[1], linewidth=2)
        
        plt.title('Cluster no: ' + str(clust.no), weight='bold')
        #plt.ylim(0, 100)
    plt.tight_layout()
    if mode=='show':
        plt.show()
    elif mode=='save':
        plt.savefig('{0}.png'.format(fname))


def compare_clusterings(clusters1, clusters2):
    '''
    Given two clusterings (i.e. lists that contains the cluster no.s for each dataseries), this method returns two comparative indices. The first index that is returned in the Rand index, whereas the second one is the Jaccard index
    
    :param clusters1: The list of cluster no.s according to the first clustering method
    :param clusters2: The list of cluster no.s according to the second clustering method
    
    :returns: Two numbers, first being the RAND index, and the second being the Jaccard index
    '''
    
    if len(clusters1) != len(clusters2):
        print("Number of members of these two clusterings are not equal")
        return 0

    if not isinstance(clusters1, np.ndarray):
        c1 = np.array(clusters1)
        c2 = np.array(clusters2)

    n = len(c1)
    c1_matrix = c1[:, np.newaxis] == c1[np.newaxis, :]
    c2_matrix = c2[:, np.newaxis] == c2[np.newaxis, :]
    
    # Extract upper triangular part to get unique pairs
    # k=1 parameter excludes the diagonal
    upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    c1_same_pairs = c1_matrix[upper_tri_mask]
    c2_same_pairs = c2_matrix[upper_tri_mask]
    
    count_same_in_both = np.sum(c1_same_pairs & c2_same_pairs)
    count_different_in_both = np.sum(~c1_same_pairs & ~c2_same_pairs)
    count_same_onlyin_c1 = np.sum(c1_same_pairs & ~c2_same_pairs)
    count_same_onlyin_c2 = np.sum(~c1_same_pairs & c2_same_pairs)
    
    rand_index = (count_same_in_both + count_different_in_both) / (count_same_in_both + count_same_onlyin_c1 + count_same_onlyin_c2 + count_different_in_both)
    jackard_index = count_same_in_both / (count_same_in_both + count_same_onlyin_c1 + count_same_onlyin_c2)
    
    return rand_index, jackard_index


def experiment_controller(inputFileName,
                          distanceMethod='pattern',
                          flatMethod='complete',
                          transform='original',
                          cMethod='maxclust',
                          cValue= 9,
                          replicate=1,
                          note='',
                          plot=True):
    """
    distanceMethod alternatives: manhattan, mse, pattern, sse, triangle
    flatMethod alternatives: 
    transform alternatives: original (no transformations), normalize and standardize
    replicate: replicates the clustering algorithm and reports the average values of rand and jaccard indexes and total operation time
        only applicable to pattern distance method
    """
    very_begin_time = time.time()

    data_w_labels, clusters_original = import_data(inputFileName, withClusters = True)
    #data_w_labels = import_data(inputFileName, withClusters = False)

    # Transformations if both are True then only normalization is performed
    if transform == 'normalize':
        data = normalize_data(data_w_labels)
    elif transform == 'standardize':
        data = standardize_data(data_w_labels)
    else:
        data = data_w_labels
    
    # If the distance method is 'pattern' replication is taken into account
    if distanceMethod=='pattern':
        run_times = []
        rands = []
        jaccards = []
        for i in range(replicate):
            begin_time = time.time()
            dist_row, cluster_list, clusters = cluster(data, distanceMethod, flatMethod, cMethod, cValue)
            run_times.append(time.time() - begin_time)
            try:
                r, j = compare_clusterings(clusters, clusters_original)
            except TypeError:
                r, j = None, None
            rands.append(r)
            jaccards.append(j)

        run_time = np.mean(run_times)
        rand = np.mean(rands)
        jaccard = np.mean(jaccards)
    # For other distance methods no replication is made
    else:
        begin_time = time.time()
        dist_row, cluster_list, clusters = cluster(data, distanceMethod, flatMethod, cMethod, cValue)
        run_time = time.time() - begin_time

        try:
            rand, jaccard = compare_clusterings(clusters, clusters_original)
        except TypeError:
            rand, jaccard = None, None

    noClusters = max(clusters)
    outputFileName = '{0}-{1}-{2}-{3}.xlsx'.format(distanceMethod,flatMethod,transform,note)
    path_adjust = os.path.join(os.getcwd(),'..','output')
    
    w = xlsxwriter.Workbook(os.path.join(path_adjust, outputFileName))
    ws = w.add_worksheet('results')
    ws.set_column('A:A', 20)
    ws.set_column('B:B', 14)
    ws.set_column('C:C', 14)
    
    # Guide: ws.write(row, col, value)
    ws.write(0,0,'File Name:')
    ws.write(0,1,inputFileName)
    ws.write(1,0,'Inter-Cluster Similarity')
    ws.write(1,1,flatMethod)
    ws.write(2,0,'cMethod')
    ws.write(2,1,cMethod)
    ws.write(3,0,'cValue')
    ws.write(3,1,cValue)
    ws.write(4,0,"Time:")
    ws.write(4,1,time.strftime("%H:%M %d/%m/%Y"))
        
    ws.write(5,0,'Distance Measure:')
    ws.write(6,0,distanceMethod)
    ws.write(7,0,distanceMethod)
    ws.write(8,0,distanceMethod)
    
    ws.write(5,2,'Metric')
    ws.write(6,2,'Jackard')
    ws.write(7,2,'Rand')
    ws.write(8,2,'Run Time')
    
    ws.write(5,4,'Transformation')
    ws.write(6,4,transform)
    ws.write(7,4,transform)
    ws.write(8,4,transform)
    
    ws.write(5,5,'Outcome')
    ws.write(6,5,jaccard)
    ws.write(7,5,rand)
    ws.write(8,5,run_time)
        
    ws.write(11,0,'Total number of clusters:')
    ws.write(11,1,noClusters)
    
    # Vectorized cluster counting
    clusters_array = np.array(clusters)
    unique_clusters, cluster_counts = np.unique(clusters_array, return_counts=True)
    
    for i, (cluster_id, count) in enumerate(zip(unique_clusters, cluster_counts)):
        ws.write(12+i,0,'Cluster {0}:'.format(cluster_id))
        ws.write(12+i,1,count)
    
    ws.write(23,0,'Index')
    ws.write(23,1,'Original Cluster')
    ws.write(24,2,'Cluster List')
    
    for i in range(len(data)):
        ws.write(24+i,0,i)
        ws.write(24+i,1,clusters_original[i])
        ws.write(24+i,2,clusters[i])
    w.close()
    
    very_end_time = time.time()
    print('Grand total time:', very_end_time - very_begin_time)
    
    if plot==True:
        plot_clusters(cluster_list, distanceMethod, mode='save', fname=os.path.join(path_adjust, os.path.splitext(outputFileName)[0]))


class Cluster(object):
    '''
    Contains information about a data-series cluster, as well as some methods to help analyzing a cluster.
    Basic attributes of a cluster (e.g. c) object are as follows;
        :ivar no: Cluster number/index
        :ivar indices: Original indices of the dataseries that are in cluster c
        :ivar sample: Original index of the dataseries that is the representative of cluster c (i.e. median element of the cluster)
        :ivar members: Members of the cluster 
        :ivar size: Number of elements (i.e. dataseries) in the cluster c
    '''

    def __init__(self, cluster_no, all_ds_indices, sample_ds, member_dss):
        self.no = cluster_no
        self.indices = all_ds_indices
        self.sample = sample_ds
        self.size = self.indices.size
        self.members = member_dss

    def error(self):
        return self.sample


'''
The main method where the user needs to specify the path to the simulation results
'''
if __name__ == '__main__':
    inputFileName = 'TestSet_wo_Osc'

    experiment_controller(inputFileName, distanceMethod='pattern_dtw', flatMethod='complete',
           transform='normalize', cMethod='maxclust', note="wSLope=6", cValue=9, replicate=1)