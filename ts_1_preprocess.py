import numpy as np
import pandas as pd
import utils

#%%
data = pd.read_hdf('data/train.h5')
SUPER_VALUES_seed = 13
#%%
# seed the random number generator to provide repeatable result
np.random.seed(SUPER_VALUES_seed)
'''
prepare testing dataset

based on the obversation on splitting between training anda testing dataset in kagglegym, testing dataset should be after training dataset in time asc.
order all data in timeseries, and use the last 1% as testing dataset.
'''

remained_sample_ids = set(data.index.tolist())
test_ids = set([])

data_gp = data.groupby('timestamp')
test_ratio = 0.01
print('create test dataset',end='')


unique_timestamp = data["timestamp"].unique()
n = len(unique_timestamp)
test_start_i = int(n*(1-test_ratio))
timesplit = unique_timestamp[test_start_i]

print('timesplit:',timesplit)
train_data  = data[data.timestamp < timesplit]
test_data = data[data.timestamp >= timesplit]

utils.save_variable(test_data,'E:/two-sigma/output/timeseries/test_data')

#%%
'''
generate timeseries chunks
'''
ts_groups = train_data.groupby('timestamp')
for key in ts_groups.groups.keys():
    print('Timestamp:',key,end=',')
    row_ids = ts_groups.groups[key]
    gp_data = train_data.ix[row_ids]
    utils.save_variable(gp_data,'E:/two-sigma/output/timeseries/tr_chunks/'+str(key))
    print('SIZE:',gp_data.shape)
