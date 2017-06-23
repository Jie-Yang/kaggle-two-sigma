import numpy as np
import pandas as pd
import random
import time as t
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import mutual_info_regression
#%%
data = pd.read_hdf('data/train.h5')


#%%
SUPER_VALUES_selected_feature_ids = [0, 88, 91, 99, 100]
SUPER_VALUES_seeds = [13,17,97,23,111]
SUPER_VALUES_seed = 13
SUPER_VALUES_kf_k=5
SUPER_VALUES_alpha=1e-07

#%%
from sklearn.metrics import r2_score
def cal_r(y_true, y_pred):
    #https://www.kaggle.com/c/two-sigma-financial-modeling/details/evaluation
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2)*np.sqrt(np.absolute(r2))
    # r in kaggle platform will be clipped at -1
    return r;
# this list is calculated based 3_model_1L_gaussian_kbest

#%%


t_preprocessing = t.time()
'''
prepare testing dataset

about 1% of full dataset: get random samples (1%) at every timestamp
'''


remained_sample_ids = set(data.index.tolist())
test_ids = set([])

data_gp = data.groupby('timestamp')
test_ratio = 0.01
for name, group in data_gp:
    print('T'+str(name),end=',')
    gp_idx = set(data_gp.groups[name])
    idx_pool = gp_idx.intersection(remained_sample_ids)
    sel_ids_len = int(len(gp_idx)*test_ratio)
    sel_ids = set(random.sample(idx_pool,sel_ids_len))
    print('sel',len(sel_ids),'samples from',len(gp_idx))
    test_ids = test_ids | sel_ids
    remained_sample_ids = remained_sample_ids - sel_ids

print('testing samples:',len(test_ids))
#%%


test_data = data.ix[test_ids]

tr_ids = list(set(data.index.tolist())-set(test_ids))
train_data = data.ix[tr_ids]

#%%
'''
break training data into 100 similar chunks: well-distributed timestamps and object id

1. there are about 1812 timestamps
2. group sample by timestamp. At each timestamp, break samples into 100 chunks randomly. At the end, pick one chunk at each timestamp, and combine into a big chunk. 
'''

#%%
data_gp = train_data.groupby('timestamp')

tr_chunk_ids = {}
for i in range(100):
    tr_chunk_ids[i] = []

for name, group in data_gp:
    print('create tr chunks from timestamp:',name)
    ids = group.index
    ids_shuffled = np.random.permutation(ids)
    sub_chunks = np.array_split(ids_shuffled,100)
    # shuffle the chunk list to avoid all small chunks always sit at the end of the list
    np.random.shuffle(sub_chunks)
    for i in range(100):
        tr_chunk_ids[i].extend(sub_chunks[i])


#%%
'''
The final output of preprocessing is:
    train_data
    tr_chunk_ids
    test_data
COST TOTAL TIME: 101.55415511131287
'''

#%%
'''
####################################################################################
MODELING

process each chunk indepedently.

'''
import copy

y_min = np.min(data['y'])
y_max = np.max(data['y'])

models_1L = []
t_accumulated = t.time()
for chunk_i in range(100):
    t_chunk_ini = t.time()
    print('C'+str(chunk_i),end=',')
    ids = tr_chunk_ids[chunk_i]
    tr_chunk = train_data.ix[ids]
    '''
     1. Remove samples with saturated values
    '''
    tr_chunk = tr_chunk.loc[~tr_chunk['y'].isin([y_min,y_max])]
    
    model_XY = tr_chunk
    tr_X = model_XY.drop(['id','y'], 1)
    tr_Y = model_XY['y'].values
    test_X = test_data.drop(['id','y'], 1)
    test_Y = test_data['y'].values

    '''
     2. replace NaN with median value
     use SUPER_VALUES_selected_feature_ids to accelerate processing
    '''
    model_X = tr_X[SUPER_VALUES_selected_feature_ids]
    imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    imputer.fit(model_X)
    
    tr_X_imputed = imputer.transform(model_X)
    test_X_imputed = imputer.transform(test_X[SUPER_VALUES_selected_feature_ids])
    #%
    '''
     3. normalization
    '''
    normalization = preprocessing.StandardScaler().fit(tr_X_imputed)
    tr_X_norm = normalization.transform(tr_X_imputed)
    test_X_norm = normalization.transform(test_X_imputed)
    
    '''
    4. kbest for over-fitting problem
    '''
    kbest =  SelectKBest(mutual_info_regression, k=1)
    kbest.fit(tr_X_norm, tr_Y)
    tr_X_kbest = kbest.transform(tr_X_norm)
    test_X_kbest = kbest.transform(test_X_norm)
    
    '''
    5. Apply regression
    '''

    chunk_tr_X = tr_X_kbest
    chunk_tr_Y = tr_Y

    skf = KFold(n_splits=SUPER_VALUES_kf_k,shuffle=True,random_state=SUPER_VALUES_seed)

    kf_i = 0
    for tr_idx, val_idx in skf.split(chunk_tr_X):
        t1 = t.time()
        print('KF'+str(kf_i),end=',')
        kf_tr_X, kf_tr_Y = chunk_tr_X[tr_idx,:], chunk_tr_Y[tr_idx]
        kf_val_X, kf_val_Y = chunk_tr_X[val_idx,:], chunk_tr_Y[val_idx]
    
        model = GaussianProcessRegressor(alpha=SUPER_VALUES_alpha,
                  random_state=SUPER_VALUES_seed)
        
        reg_t_0 = t.time()
        model.fit(kf_tr_X, kf_tr_Y)
        kf_tr_Y_pred = model.predict(kf_tr_X)
        kf_tr_r = cal_r(kf_tr_Y,kf_tr_Y_pred)
        print('tr:'+str(kf_tr_r),end=',')
        
        kf_val_Y_pred = model.predict(kf_val_X)
        kf_val_r = cal_r(kf_val_Y,kf_val_Y_pred)
        print('val:'+str(kf_val_r),end=',')
        
        if kf_val_r <=0:
            print('(DISCARD)',end=',')
        else:
            model_test_Y_pred = model.predict(test_X_kbest)
            kf_test_r = cal_r(test_Y, model_test_Y_pred)
            print('test:'+str(kf_test_r),end=',')
            
            if kf_test_r > 0:
                models_1L.append(copy.deepcopy(model))
                print('(SAVED',str(len(models_1L)),')',end=',')
            else:
                print('(DISCARD)',end=',')

        t_chunk_finish = t.time()
        print('cost:'+str(int(t_chunk_finish-t_chunk_ini))+'sec/'+str(int((t_chunk_finish-t_accumulated)/60))+'min')
        kf_i +=1
        break

    # filter model
    
    
    chunk_i += 1
    #print('regression cost:',t.time()-t0)

    #print('############################')
        
#%%

model_i = 0
test_Y_sum = np.zeros(len(test_Y))
for model in models_1L:
    model_test_Y_pred = model.predict(test_X_kbest)
    test_Y_sum += model_test_Y_pred
    print(model_i,'test:',cal_r(test_Y, model_test_Y_pred))
    model_i +=1
    
print('AVERAGE test:',cal_r(test_Y, test_Y_sum/len(models_1L)))   
#%%
'''
alpha_test: 1e-10-->tr: 0.0359921585282,val: 0.0144652632601,test: -0.0111363913835
alpha_test: 1e-09-->tr: 0.0359160740403,val: 0.0160476551141,test: -0.0134063128865
alpha_test: 1e-08-->tr: 0.0357893683734,val: 0.0189098512741,test: -0.00969781321665
alpha_test: 1e-07-->tr: 0.034818412793,val: 0.0207464996291,test: 0.011056320345  ---> BEST
alpha_test: 1e-06-->tr: 0.0312144317061,val: 0.00850754154522,test: 0.016458037367
alpha_test: 9.999999999999999e-06-->tr: 0.0279351596482,val: -0.0150461118187,test: 0.0156451135145
alpha_test: 9.999999999999999e-05-->tr: 0.0268936515551,val: -0.010534349974,test: 0.0155052528318
'''