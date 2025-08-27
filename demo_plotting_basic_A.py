import numpy as np
from simclstr.plotting import interactive_plot_clusters, multiple_tabs_interactive_plot_clusters, plot_clusters
from simclstr.clusterer import read_time_series, perform_clustering
import os

def main():
    model_path_vensim = os.path.expanduser('~/Desktop/SESDYN_Basic_A/Basic_A_Vensim.xlsx')

    read_data = read_time_series(model_path_vensim)

    clustering_results = perform_clustering(read_data, distance='pattern_dtw', cMethod='maxclust', cValue = 6, plotDendrogram=True)

    multiple_tabs_interactive_plot_clusters(clustering_results[1], 'pattern_dtw')

if __name__ == "__main__":
    main()