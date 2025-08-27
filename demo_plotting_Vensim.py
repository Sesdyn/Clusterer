import numpy as np
from simclstr.plotting import interactive_plot_clusters, multiple_tabs_interactive_plot_clusters, plot_clusters
from simclstr.clusterer import read_time_series, perform_clustering, simulate_from_vensim
import os

def main():
    model_path = os.path.expanduser('~/Desktop/temperature model/temperature model.mdl')

    parameter_set = {'Initial actual temperature': [15, 20, 25],
                    'Initial measured temperature': [15, 20, 25],
                    'Desired temperature': [15, 20, 25],
                    'Adjustment time': [3*i for i in range(1, 6)],
                    'Measurement delay': [3*i for i in range(1, 6)]
                    }

    output_of_interest = 'Actual Temperature'

    simulation_results = simulate_from_vensim(model_path, parameter_set, output_of_interest)

    clustering_results = perform_clustering(simulation_results, distance='pattern', cMethod='maxclust', cValue = 6, plotDendrogram=True)

    multiple_tabs_interactive_plot_clusters(clustering_results[1], 'pattern')


if __name__ == "__main__":
    main()