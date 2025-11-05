# Scipy 3000'den fazla zaman serisi olduğu zaman max recursion depth hatası veriyor.
# Örnek olarak kalması için Arms Race modelini buraya ekledim, makul sayıda zaman serisi için hızlı bir şekilde çalışıyor.

import numpy as np
from simclstr.plotting import interactive_plot_clusters, multiple_tabs_interactive_plot_clusters, plot_clusters
from simclstr.clusterer import read_time_series, perform_clustering, simulate_from_vensim
from print_distances import print_distances
import os

def main():
    model_path = os.path.expanduser('data_files/arms race.mdl')

    parameter_set = {"initial arms expenditure A": [200, 400],
                    "initial arms expenditure B": [200, 400],
                    "fear responsiveness A": [1, 4],
                    "fear responsiveness B": [1, 4],
                    "grievance A": [0, 5],
                    "grievance B": [0, 5],
                    "restraint A": [1, 4],
                    "restraint B": [1, 4],
                    "perception delay time of A": [1, 4],
                    "perception delay time of B": [1, 4]
                    }

    output_of_interest = 'arms expenditure A'

    simulation_results = simulate_from_vensim(model_path, parameter_set, output_of_interest)

    clustering_results = perform_clustering(simulation_results, distance='pattern_wdtw', cMethod='maxclust', cValue = 6, plotDendrogram=True, transform='normalize')

    multiple_tabs_interactive_plot_clusters(clustering_results[1], 'pattern_wdtw')

    print_distances(clustering_results[0], [307, 528, 988, 786, 1011, 1008])

    submatrix = [time_series for time_series in simulation_results if time_series.index in [307, 528, 988, 786, 1011, 1008]]
    
    submatrix_clustering_results = perform_clustering(submatrix, distance='pattern_wdtw', cMethod='maxclust', cValue = 1, plotDendrogram=True, transform='normalize')

    multiple_tabs_interactive_plot_clusters(submatrix_clustering_results[1], 'pattern_wdtw')

if __name__ == "__main__":
    main()