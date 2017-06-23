# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:15:58 2017

@author: jyang
"""

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
import utils
import numpy as np

from ts_Model_Classes import Model1L


        
#%%
print('-----------------init first Model_2nd----------------')
SUPER_start_timestamp = 10
SUPER_window_size = SUPER_start_timestamp
SUPER_kf_k = 5
SUPER_seed = 13
SUPER_alpha = 1e-8
skf = KFold(n_splits=SUPER_kf_k,shuffle=True,random_state=SUPER_seed)

'''
load lag_models (FIFO)
'''
lag_models = []
for lag_model_timestamp in range(SUPER_start_timestamp-SUPER_window_size,SUPER_start_timestamp,1):
    lag_model = utils.read_variable('E:/two-sigma/output/timeseries/model_1L/'+str(lag_model_timestamp))
    lag_models.append(lag_model)

'''
Init Model_2L
'''
timestamp = SUPER_start_timestamp
    
print('processing',timestamp)

tr_chunk = utils.read_variable('E:/two-sigma/output/timeseries/tr_chunks/'+str(timestamp))

'''
CV
'''
kf_i = 0
best_model_2L_r = 0
for tr_idx, val_idx in skf.split(tr_chunk):
    print(timestamp,'kf'+str(kf_i),end=',')
    kf_i += 1
    
    kf_tr_chunk = tr_chunk.iloc[tr_idx]

    '''
    training
    '''
    lag_X = np.zeros([len(kf_tr_chunk.y),len(lag_models)])
    for lag_model_i, lag_model in enumerate(lag_models):
        y_pred = lag_model.predict(kf_tr_chunk)
        lag_X[:,lag_model_i] = y_pred
        #print('-->',lag_model_i,utils.cal_r(kf_tr_chunk.y,y_pred))

   

    model_2L = GaussianProcessRegressor(alpha=SUPER_alpha,
              random_state=SUPER_seed)
    
    model_2L.fit(lag_X, kf_tr_chunk.y)
    
    tr_Y_pred = model_2L.predict(lag_X)
    tr_r = utils.cal_r(kf_tr_chunk.y, tr_Y_pred)
    print('tr:',tr_r,end=',')
    
    '''
    Val
    '''
    kf_val_chunk = tr_chunk.iloc[val_idx]
    lag_X = np.zeros([len(kf_val_chunk.y),len(lag_models)])
    for lag_model_i, lag_model in enumerate(lag_models):
        y_pred = lag_model.predict(kf_val_chunk)
        lag_X[:,lag_model_i] = y_pred
    val_Y_pred = model_2L.predict(lag_X)
    val_r = utils.cal_r(kf_val_chunk.y, val_Y_pred)
    print('val:',val_r,end='')
    
    if val_r > best_model_2L_r:
        best_model_2L = model_2L
        best_model_2L_r = val_r
        print('-->(SELECTED)')
    else:
        print('')
        
'''
create a new t_model based on tr_X and y_pred ( there is no true Y in testing)
'''
lag_X = np.zeros([len(tr_chunk.y),len(lag_models)])
for lag_model_i, lag_model in enumerate(lag_models):
    y_pred = lag_model.predict(tr_chunk)
    lag_X[:,lag_model_i] = y_pred
best_mdoel_2L_y_pred =  best_model_2L.predict(lag_X)

tr_chunk.y =  best_mdoel_2L_y_pred

new_t_model = Model1L(SUPER_seed, alpha=5, kbest_k=10)
new_r = new_t_model.fit(tr_chunk)
print('new T model:',new_r)
'''
Update lag model pool (FIFO)
'''
lag_models.pop(0)
lag_models.append(new_t_model)
        
      
        
    
#%%
test_timestamps = []
test_r2 = []
for test_timestamp in range(timestamp+1, 1794,1):
    print('-----------------Predict & Re-train----------------')
    print('[predict',test_timestamp,']')
    tr_chunk = utils.read_variable('E:/two-sigma/output/timeseries/tr_chunks/'+str(test_timestamp))
    
    '''
    predict
    '''
    tr_chunk_y_true = tr_chunk.y
    lag_X = np.zeros([len(tr_chunk.y),len(lag_models)])
    for lag_model_i, lag_model in enumerate(lag_models):
        y_pred = lag_model.predict(tr_chunk)
        print('  L'+str(lag_model_i),utils.cal_r(tr_chunk_y_true,y_pred))
        lag_X[:,lag_model_i] = y_pred
    best_model_2L_y_pred =  best_model_2L.predict(lag_X)
    test_r = utils.cal_r(tr_chunk_y_true,best_model_2L_y_pred)
    print('                        predict:',test_r)
    
    
    test_timestamps.append(test_timestamp)
    test_r2.append(test_r)
    
    '''
    Re-train
    '''
    print('[Re-train]')

    '''
    update t_model pool
    '''
    #tr_chunk.y =  best_model_2L_y_pred
    
    new_t_model = Model1L(SUPER_seed, alpha=5, kbest_k=10)
    new_r = new_t_model.fit(tr_chunk)
    print('  new 1L model:',new_r)
    
    '''
    Update lag model pool (FIFO)
    '''

    lag_models.pop(0)
    lag_models.append(new_t_model)
    
    
    '''
    update best_model_2L
    '''
    # update lag_X
    lag_X = np.zeros([len(tr_chunk.y),len(lag_models)])
    for lag_model_i, lag_model in enumerate(lag_models):
        y_pred = lag_model.predict(tr_chunk)
        #print('  L'+str(lag_model_i),utils.cal_r(tr_chunk_y_true,y_pred))
        lag_X[:,lag_model_i] = y_pred
    best_model_2L = GaussianProcessRegressor(alpha=SUPER_alpha,
              random_state=SUPER_seed)
    '''
    How to used "Reward" instead of true Y values from KaggleGYM in best_model_2L updating process?
    '''
    best_model_2L.fit(lag_X, tr_chunk_y_true)
    best_model_2L_y_pred =  best_model_2L.predict(lag_X)
    retrain_r = utils.cal_r(tr_chunk_y_true,best_model_2L_y_pred)
    print('  retrain:',retrain_r)
    



#%%
import matplotlib.pyplot as plt

plt.plot(test_timestamps,test_r2,'g-o')
plt.xlabel('time')
plt.ylabel('r2')
plt.title('Window Size:'+str(SUPER_start_timestamp))
plt.show()


