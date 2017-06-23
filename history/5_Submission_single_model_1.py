import kagglegym
import numpy as np
import random
import time as t
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import mutual_info_regression

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

data = observation.train


#%%
SUPER_VALUES_selected_feature_ids = [0, 88, 91, 99, 100]
SUPER_VALUES_seed = 13
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
class Model1L:
    def __init__(self, y_min, y_max, seed, alpha, selected_feature_ids):
        self.y_saturated_values = [y_min, y_max]
        self.y_min = y_min
        self.y_max = y_max
        self.alpha = alpha
        self.seed = seed
        self.selected_feature_ids = selected_feature_ids

    def fit(self, tr_chunk):
        tr_Y = tr_chunk['y'].values
        tr_X = tr_chunk.drop(['id','y'], 1)
        '''
         1. Remove samples with saturated values
        '''
        tr_chunk = tr_chunk.loc[~tr_chunk['y'].isin(self.y_saturated_values)]
        tr_X = tr_chunk.drop(['id','y'], 1)
        tr_Y = tr_chunk['y'].values
        '''
        use predefined small set of feature to save time
        '''
        tr_X = tr_X[self.selected_feature_ids]
        '''
         2. replace NaN with median value
         use SUPER_VALUES_selected_feature_ids to accelerate processing
        '''
        self.imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
        self.imputer.fit(tr_X)
        tr_X = self.imputer.transform(tr_X)
        #%
        '''
         3. normalization
        '''
        self.normalization = preprocessing.StandardScaler().fit(tr_X)
        tr_X= self.normalization.transform(tr_X)
        '''
        4. kbest for over-fitting problem
        '''
        self.kbest =  SelectKBest(mutual_info_regression, k=1)
        self.kbest.fit(tr_X, tr_Y)
        tr_X = self.kbest.transform(tr_X)
        
        '''
        5. Apply regression
        '''
        self.model = GaussianProcessRegressor(alpha=SUPER_VALUES_alpha,
                  random_state=SUPER_VALUES_seed)
        
        self.model.fit(tr_X, tr_Y)
        
        tr_Y_pred = self.model.predict(tr_X)
        
        '''
        6. set value to min or max if it is out of scope
        '''
        tr_Y_pred[tr_Y_pred<self.y_min] = self.y_min
        tr_Y_pred[tr_Y_pred>self.y_max] = self.y_max        
        
        tr_r = cal_r(tr_Y,tr_Y_pred)

        return tr_r

    def predict(self,X):
        
        if 'id' in X: X = X.drop('id', 1)        
        has_Y = False
        if 'y' in X:
            has_Y = True
            Y = X['y'].values
            X = X.drop('y', 1)        
        X = X[self.selected_feature_ids]
        X = self.imputer.transform(X)
        X = self.normalization.transform(X)
        X = self.kbest.transform(X)
        Y_pred = self.model.predict(X)
        Y_pred[Y_pred<self.y_min] = self.y_min
        Y_pred[Y_pred>self.y_max] = self.y_max    
        
        if has_Y:
            return cal_r(Y,Y_pred)
        else:
            return Y_pred
#%%


t_preprocessing = t.time()

# seed the random number generator to provide repeatable result
np.random.seed(SUPER_VALUES_seed)
'''
prepare testing dataset

about 1% of full dataset: get random samples (1%) at every timestamp
'''

remained_sample_ids = set(data.index.tolist())
test_ids = set([])

data_gp = data.groupby('timestamp')
test_ratio = 0.01
print('create test dataset',end='')
for name, group in data_gp:
    #print('T'+str(name),end=',')
    gp_idx = set(data_gp.groups[name])
    idx_pool = gp_idx.intersection(remained_sample_ids)
    sel_ids_len = int(len(gp_idx)*test_ratio)
    sel_ids = set(random.sample(idx_pool,sel_ids_len))
    #print('sel',len(sel_ids),'samples from',len(gp_idx))
    print('.',end='')
    test_ids = test_ids | sel_ids
    remained_sample_ids = remained_sample_ids - sel_ids

print('(DONE)',len(test_ids))
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

print('create chunks',end='')
for name, group in data_gp:
    #print('create tr chunks from timestamp:',name)
    print('.',end='')
    ids = group.index
    ids_shuffled = np.random.permutation(ids)
    sub_chunks = np.array_split(ids_shuffled,100)
    # shuffle the chunk list to avoid all small chunks always sit at the end of the list
    np.random.shuffle(sub_chunks)
    for i in range(100):
        tr_chunk_ids[i].extend(sub_chunks[i])

print('(DONE)')

#%%
'''
MODELING
'''
import copy

y_min = np.min(data['y'])
y_max = np.max(data['y'])

t_total_start = t.time()

good_models = []
for chunk_i in range(len(tr_chunk_ids)):
    t_chunk_start = t.time()
    print('C'+str(chunk_i),end=',')
    tr_ids = tr_chunk_ids[chunk_i]
    chunk = train_data.ix[tr_ids]

    model = Model1L(y_min, y_max, SUPER_VALUES_seed, SUPER_VALUES_alpha, SUPER_VALUES_selected_feature_ids)
    
    tr_r = model.fit(chunk)
    print('tr:'+str(tr_r),end=',')
  
    val_chunk_i = chunk_i+1
    val2_chunk_i = chunk_i+2
    if chunk_i == len(tr_chunk_ids)-1:
        val_chunk_i = chunk_i-1
        val2_chunk_i = chunk_i-2
    elif chunk_i == len(tr_chunk_ids)-2:
        val2_chunk_i = chunk_i-1
    val_ids = tr_chunk_ids[val_chunk_i]
    chunk = train_data.ix[val_ids]
    val_r = model.predict(chunk)
    print('val:'+str(val_r),end=',')
    
    if val_r <=0:
        print('(DISCARD)',end=',')
    else:
        val2_ids = tr_chunk_ids[val2_chunk_i]
        chunk = train_data.ix[val2_ids]
        val2_r = model.predict(chunk)
        print('val2:'+str(val2_r),end=',')
        if val2_r <=0:
            print('(DISCARD)',end=',')
        else:
            final_model = model
            print('(SAVED)',end=',')
            test_r= model.predict(test_data)
            print('test:'+str(test_r))
            
            good_models.append(copy.deepcopy(model))

    t_chunk_finish = t.time()
    print(str(int(t_chunk_finish-t_chunk_start))+'sec,'+str(int((t_chunk_finish-t_total_start)/60))+'min')

    
#%%
'''
predict target
'''
#processed_test_c = 0
#while True:
#    target = observation.target
#    features = observation.features
#
#    target.y = final_model.predict(features)
#    
#    processed_test_c += len(target.y)
#
#    observation, reward, done, info = env.step(target)
#    print(processed_test_c,'reward:',reward)
#    if done:
#        print("Public score: {}".format(info["public_score"]))
#        break

#%%
'''
predict target
'''

#env = kagglegym.make()
#observation = env.reset()


processed_test_c = 0
while True:
    target = observation.target
    features = observation.features

    y_pred_sum = np.zeros(len(target.y))
    for model in good_models:
        y_pred_sum += model.predict(features)
    y_pred = y_pred_sum / len(good_models)
    target.y = y_pred
    
    processed_test_c += len(target.y)

    observation, reward, done, info = env.step(target)
    print(processed_test_c,'reward:',reward)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
    
#%%

val2_ids = tr_chunk_ids[val2_chunk_i]
chunk = train_data.ix[val2_ids]

chunk_X = chunk.drop('y', 1)
chunk_Y = chunk['y']

y_pred_sum = np.zeros(len(chunk_Y))
for model in good_models:
    chunk_Y_pred = model.predict(chunk_X)
    y_pred_sum += chunk_Y_pred
    
    print(cal_r(chunk_Y,chunk_Y_pred))
y_pred = y_pred_sum / len(good_models)
print('aver:',cal_r(chunk_Y,y_pred))