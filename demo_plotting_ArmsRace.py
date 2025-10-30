# Scipy 3000'den fazla zaman serisi olduğu zaman max recursion depth hatası veriyor.
# Örnek olarak kalması için Arms Race modelini buraya ekledim, makul sayıda zaman serisi için hızlı bir şekilde çalışıyor.

import numpy as np
from simclstr.plotting import interactive_plot_clusters, multiple_tabs_interactive_plot_clusters, plot_clusters
from simclstr.clusterer import read_time_series, perform_clustering, simulate_from_vensim
import os

def main():
    model_path = os.path.expanduser('Data Files/arms race.mdl')

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

    clustering_results = perform_clustering(simulation_results, distance='pattern_dtw', cMethod='maxclust', cValue = 6, plotDendrogram=True, transform='normalize')

    from scipy.spatial.distance import squareform
    import pandas as pd

    dist_matrix = squareform(clustering_results[0])

    pd.DataFrame(dist_matrix).to_csv("dist_matrix.csv", index=False, header=False)

    multiple_tabs_interactive_plot_clusters(clustering_results[1], 'pattern_dtw')

if __name__ == "__main__":
    main()