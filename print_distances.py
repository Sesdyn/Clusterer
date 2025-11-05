import pandas as pd

csv_path = 'data_files/Arms_Race_dist_matrix.csv'
df = pd.read_csv(csv_path)

distances = [
[0, 1],
[0, 2]
]

for each_distance in distances:
    i = each_distance[0]
    j = each_distance[1]
    distance_value = df.iloc[i, j]
    print(f"Distance between series {i} and {j}: {distance_value}")