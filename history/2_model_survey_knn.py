import numpy as np
import pandas as pd
import utils

#%%

chunk_i = 45
tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))

test_chunk = utils.read_variable('output/test_data')

tr_Y = tr_chunk['y']


#%%
'''
Result:
    n_neighbors = 5
    tr: 0.477527312509,val: -0.467288713014
    cost: traning 1.1630001068115234 ,prediction: 120.34500002861023
    
    n_neighbors = 10
    tr: 0.338104646376,val: -0.346948597567
    cost: traning 1.1529998779296875 ,prediction: 123.15199995040894
    
'''
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import time
from sklearn.model_selection import KFold


Y = tr_chunk['y'].values
X = tr_chunk.drop(['id','y'], 1).as_matrix()

skf = KFold(n_splits=5,shuffle=True,random_state=13)
for tr_idx, val_idx in skf.split(X):
    
    tr_X, tr_Y = X[tr_idx,:], Y[tr_idx]
    val_X, val_Y = X[val_idx,:], Y[val_idx]

    imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    imputer.fit(tr_X)
    tr_X_imputed = imputer.transform(tr_X)
    val_X_imputed = imputer.transform(val_X)
    normalization = preprocessing.StandardScaler().fit(tr_X_imputed)
    tr_X_norm = normalization.transform(tr_X_imputed)
    val_X_norm = normalization.transform(val_X_imputed)


    '''
        n_neighbors : int, optional (default = 5)
    '''
    model = KNeighborsRegressor(n_neighbors = 10)
    
    t0 = time.time()
    model.fit(tr_X_norm, tr_Y)
    t1 = time.time()
    
    tr_Y_pred = model.predict(tr_X_norm)
    t2 = time.time()
    
    tr_r = utils.cal_r(tr_Y,tr_Y_pred)
    print('tr:',tr_r,end=',')
    
    val_Y_pred = model.predict(val_X_norm)
    val_r = utils.cal_r(val_Y,val_Y_pred)
    print('val:',val_r)
    
    print('cost: traning',t1-t0,',prediction:',t2-t1)        
        

    break