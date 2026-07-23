from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np


def print_distances(dRow, indices_of_ts):
    distance_matrix = squareform(dRow)

    n = len(indices_of_ts)
    columns = [''] + [str(idx) for idx in indices_of_ts]
    
    df_data = np.zeros((n + 1, n + 1), dtype=object)
    
    df_data[0, 0] = ''
    df_data[0, 1:] = [str(idx) for idx in indices_of_ts]
    
    df_data[1:, 0] = [str(idx) for idx in indices_of_ts]

    for i in range(n):
        for j in range(n):
            df_data[i + 1, j + 1] = distance_matrix[indices_of_ts[i], indices_of_ts[j]]
        
    df = pd.DataFrame(df_data, columns=columns, index=range(n + 1))
    
    print(df.to_string(index=False))