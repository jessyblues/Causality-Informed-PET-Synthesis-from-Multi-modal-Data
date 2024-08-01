import numpy as np
import os
import pandas as pd
import pickle

PET_kind = 'AV45'
training_csv = f'./T1toPET/unet/config/pair_t1_{PET_kind}_training_with_csf.csv'
need_values = ['TAU','PTAU','Age','PTEDUCAT'] if PET_kind == 'AV1451' else ['ABETA','Age','PTEDUCAT']

array = pd.read_csv(training_csv)
min_and_max = {}
for k in need_values:
    min_and_max[k] = [np.min(array[k]), np.max(array[k])]
    print(k, min_and_max[k])
    
with open(f'/home1/yujiali/T1toPET/unet/config/{PET_kind}_min_and_max.pkl', 'wb') as file:
    pickle.dump(min_and_max, file)