import clusterer as cl
import pandas as pd
import behavior_splitter as bs

LNRGR = cl.import_pandas_data("StellaDefault", "LNRGR")
data = LNRGR.iloc[:,1:len(LNRGR.columns)-1]

fv=bs.construct_features(data.as_matrix())



