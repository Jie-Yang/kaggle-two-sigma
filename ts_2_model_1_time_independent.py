import numpy as np
import os
import utils
from sklearn.model_selection import KFold
import time as t
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import mutual_info_regression
from utils import cal_r
from ts_Model_Classes import Model1L

#%%
test_chunk = utils.read_variable('E:/two-sigma/output/timeseries/test_data')

#%%

#%%

SUPER_kf_k = 10
SUPER_seed = 13
SUPER_kbest_k = 10
SUPER_alpha = 5
skf = KFold(n_splits=SUPER_kf_k,shuffle=True,random_state=SUPER_seed)

for root, dirs, files in os.walk('E:/two-sigma/output/timeseries/tr_chunks'):
    for timestamp in files:
        
        model_path = 'E:/two-sigma/output/timeseries/model_1L/'+str(timestamp)
        if os.path.isfile(model_path):
            print(timestamp,'Already Exist.')
        else:
            chunk_path = os.path.join(root, str(timestamp))
            tr_chunk = utils.read_variable(chunk_path)
            #print(tr_chunk.info())
    
            kf_i = 0
            best_val_r = 0
            
            for tr_idx, val_idx in skf.split(tr_chunk):
                print(timestamp,'kf'+str(kf_i),end=',')
            
                kf_i += 1
                
                kf_tr = tr_chunk.iloc[tr_idx]
                model = Model1L(SUPER_seed, SUPER_alpha, SUPER_kbest_k)
                tr_r = model.fit(kf_tr)
                print('tr:',tr_r,end=',')
                '''
                val
                '''
                kf_val = tr_chunk.iloc[val_idx]
                val_r =  model.val(kf_val)
                print('val:',val_r,end='') 
                
                if val_r > best_val_r:
                    best_val_r = val_r 
                    best_model = model
                    print('-->(SELECTED)')
                else:
                    print('')
            
            overall_r = best_model.val(tr_chunk)
            print(timestamp,'Total:',overall_r)
            test_r = best_model.val(test_chunk)
            print(timestamp,'Test:',test_r)
            
            '''
            persist the best model
            '''
            utils.save_variable(best_model,model_path)
        print('--------------------------------------------')
    
    
    
    