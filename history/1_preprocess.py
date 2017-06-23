import numpy as np
import pandas as pd
import utils

#%%
data = pd.read_hdf('data/train.h5')

#%%
'''
prepare testing dataset

about 1% of full dataset: get random samples (1%) at every timestamp
'''
import random

remained_sample_ids = set(data.index.tolist())
sel_sample_ids = set([])

data_gp = data.groupby('timestamp')
test_ratio = 0.01
for name, group in data_gp:
    print('T'+str(name),end=',')
    gp_idx = set(data_gp.groups[name])
    idx_pool = gp_idx.intersection(remained_sample_ids)
    sel_ids_len = int(len(gp_idx)*test_ratio)
    sel_ids = set(random.sample(idx_pool,sel_ids_len))
    print('sel',len(sel_ids),'samples from',len(gp_idx))
    sel_sample_ids = sel_sample_ids | sel_ids
    remained_sample_ids = remained_sample_ids - sel_ids

utils.save_variable(sel_sample_ids,'output/test_ids')
print('testing samples:',len(sel_sample_ids))
del sel_sample_ids
#%%
test_ids = list(utils.read_variable('output/test_ids'))
'''
check whether testing dataset is well distributed among different obj ids
'''
data_gp = data.groupby('id')
overall_stats_by_id = {}
for name, group in data_gp:
    print(name,'give has samples:',group.shape[0])
    overall_stats_by_id[name] = group.shape[0]

test_data = data.ix[test_ids]
data_gp = test_data.groupby('id')
test_ratio_per_obj_ids = []
for name, group in data_gp:
    print(name,'give testing samples:',group.shape[0],'from',overall_stats_by_id[name])
    ratio = group.shape[0]/overall_stats_by_id[name]
    test_ratio_per_obj_ids.append(ratio)

'''
by plotting out test_ratio_per_obj_ids
Result

testing samples per obj_id is between 1% to 7% of overall sample set of each object. 
hence, we could believe that testing samples are well distributed among different obj ids
'''

utils.save_variable(test_data, 'output/test_data')
tr_ids = list(set(data.index.tolist())-set(test_ids))
train_data = data.ix[tr_ids]
utils.save_variable(train_data, 'output/train_data')
#%%
'''
break training data into 100 similar chunks: well-distributed timestamps and object id

1. there are about 1812 timestamps
2. group sample by timestamp. At each timestamp, break samples into 100 chunks randomly. At the end, pick one chunk at each timestamp, and combine into a big chunk. 
'''
train_data = utils.read_variable('output/train_data')
#%%
data_gp = train_data.groupby('timestamp')

chunk_ids = {}
for i in range(100):
    chunk_ids[i] = []

for name, group in data_gp:
    print('processing tr samples at timestamp:',name)
    ids = group.index
    ids_shuffled = np.random.permutation(ids)
    sub_chunks = np.array_split(ids_shuffled,100)
    # shuffle the chunk list to avoid all small chunks always sit at the end of the list
    np.random.shuffle(sub_chunks)
    for i in range(100):
        chunk_ids[i].extend(sub_chunks[i])



#%% validate chunks
total_chunks_nu = 0
chunk_nus = []
for i in range(100):
    total_chunks_nu += len(chunk_ids[i])
    chunk_nus.append(len(chunk_ids[i]))
    print(len(chunk_ids[i]))
    
print('does the total number of all chunks correct?',total_chunks_nu==train_data.shape[0])
# plot out chunks_nus to see whether size of each chunk is even distributed.
# observation: yes
print('min:',min(chunk_nus))
print('max:',max(chunk_nus))
'''
min: 16902
max: 16987
'''
#%% save chunk into files for further processing
for i in range(100):
    ids = chunk_ids[i]
    chunk_data = train_data.ix[ids]
    print('chunk',i,chunk_data.shape)
    utils.save_variable(chunk_data, 'output/tr_chunks/'+str(i))

#%%
'''
Remove samples with saturated values (equal to min or max value)
'''
y_min = np.min(data['y'])
y_max = np.max(data['y'])

for chunk_i in range(100):
    print('processing chunk:',chunk_i)
    tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))
    
    tr_chunk_unsaturated = tr_chunk.loc[~tr_chunk['y'].isin([y_min,y_max])]

    print('desaturate: keep',tr_chunk_unsaturated.shape[0],'from',tr_chunk.shape[0],tr_chunk_unsaturated.shape[0]/tr_chunk.shape[0])
    utils.save_variable(tr_chunk_unsaturated, 'output/tr_chunks/'+str(chunk_i))

'''
output
processing chunk: 97
desaturate: keep 16691 from 16916 0.986698983211161
processing chunk: 98
desaturate: keep 16766 from 16975 0.9876877761413844
processing chunk: 99
desaturate: keep 16744 from 16965 0.9869731800766284
'''
#%%
'''
check obj ids even distributed among chunks.
 1. there are 100 chunks
 2. there are 1424 objs (id range: 0 to 2158)

''' 
chunk_vs_obj_matrix = np.zeros((100,2159))
for chunk_id in range(100):
    print('processing chunk:',chunk_id)
    tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_id))
    obj_ids = tr_chunk['id']
    for obj_id in obj_ids:
        chunk_vs_obj_matrix[chunk_id,obj_id] = 1

#%%
import matplotlib.pyplot as plt
plt.matshow(chunk_vs_obj_matrix)
plt.show()

'''
according to the matrix plot,objs are well distributed across different chunk.
'''

#%%
'''
check y distribution of each chunk
'''
chunks_mean = np.zeros(100)
chunks_std = np.zeros(100)
chunks_min = np.zeros(100)
chunks_max = np.zeros(100)

for chunk_i in range(100):
    print('processing chunk:',chunk_i)
    tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))
    ys = tr_chunk['y']

    chunks_min[chunk_i] = np.min(ys)
    chunks_max[chunk_i] = np.max(ys)
    chunks_mean[chunk_i] = np.mean(ys)
    chunks_std[chunk_i] = np.std(ys)

#%%
xs = range(100)
plt.plot(xs, chunks_min, 'r--',
         xs, chunks_max, 'g--', 
         xs, chunks_mean, 'g^', 
         xs, chunks_std, 'rs')
plt.show()

'''
Observations

1. Y is well distributed across chunks
2. all chunks have the exactly the same MAX and MIN values.
'''



