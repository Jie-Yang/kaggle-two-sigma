import numpy as np
import pandas as pd
import utils
from ChunkTree import ChunkTree

#%%

chunk_i = 45
tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))

test_chunk = utils.read_variable('output/test_data')

tr_Y = tr_chunk['y']

#%%
'''
Gaussian process regression 
'''

from ChunkGaussian import ChunkGaussian

model = ChunkGaussian(kf_k=5, seed=13,alpha=1e-10, random_state=13)
model.fit(tr_chunk, test_chunk)

'''
output 

GOOD: Gaussian have the capability to describe data well even with over-fitting problem
Other models do not provide such high R in training dataset.

CAUTION: Gaussian Regress is relatively slow: 
    fit: 2 mins
    pred: 1.5 mins

alpha=1e-10, random_state=13
KF 0,tr: 0.999992382853,val: -0.124889191349,test: -0.105056916002 (SAVED)
KF 1,tr: 0.999992346584,val: -0.150348260419,test: -0.121252752459 (SAVED)
KF 2,tr: 1.0,val: -0.156077356393,test: -0.126744712588 (SAVED)
KF 3,tr: 0.999992406752,val: -0.131700525879,test: -0.12563532255 (SAVED)
KF 4,tr: 1.0,val: -0.128457457831,test: -0.120311488206 (SAVED)
AVERAGE tr: 0.979399459497,test: -0.105508681855
'''

#%%
'''
good training R, but bad val and testing. Try to solve the over-fitting problem

bigger alpha could solve over-fitting problem.
'''
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import time

'''
Larger values correspond to increased noise level in the observations and reduce potential numerical issue during fitting.
'''
alpha = 100 # default 1e-10
for i in range(12):
    print('alpha',alpha)
    model = GaussianProcessRegressor(alpha=alpha,random_state=13)
    
    model_Y = tr_chunk['y'].values
    model_X = tr_chunk.drop(['id','y'], 1)
    test_Y = test_chunk['y'].values
    test_X = test_chunk.drop(['id','y'], 1)
    X = model_X
    imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
    imputer.fit(X)
    model_X_imputed = imputer.transform(X)
    test_X_imputed = imputer.transform(test_X)
    normalization = preprocessing.StandardScaler().fit(model_X_imputed)
    model_X_norm = normalization.transform(model_X_imputed)
    
    test_X_norm = normalization.transform(test_X_imputed)
    
    t0 = time.time()
    model.fit(model_X_norm, model_Y)
    t1 = time.time()
    
    tr_Y_pred = model.predict(model_X_norm)
    t2 = time.time()
    
    tr_r = utils.cal_r(tr_Y,tr_Y_pred)
    print('tr:',tr_r,end=',')
    
    test_Y_pred = model.predict(test_X_norm)
    test_r = utils.cal_r(test_Y,test_Y_pred)
    print('test:',test_r)
    
    print('cost: traning',t1-t0,',prediction:',t2-t1)
    
    alpha *=10
    
#%%
'''
alpha 0.1
tr: 0.995232505044,test: -0.115465678033
alpha 1.0
tr: 0.863627978178,test: -0.0694567034969
alpha 10.0
tr: 0.416104407169,test: -0.0203804800129

alpha 1000
tr: 0.01, test -???
'''
from sklearn.gaussian_process import GaussianProcessRegressor
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

    alpha = 1e-1 # default 1e-10
    for i in range(3):
        print('alpha',alpha)
        model = GaussianProcessRegressor(alpha=alpha,random_state=13)
        
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
        
        alpha *=10

    break

'''
Result: over-fitting problem can NOT be saved by adjust parameter ALPHA.

alpha 0.001
tr: 0.999991765969,val: -0.124380256409
cost: traning 69.26600003242493 ,prediction: 23.55999994277954
alpha 0.01
tr: 0.999933032028,val: -0.123202201746
cost: traning 74.13499999046326 ,prediction: 23.84999990463257
alpha 0.1
tr: 0.995424706817,val: -0.112963387477
cost: traning 70.9319999217987 ,prediction: 23.822999954223633
alpha 1
tr: 0.864216139075,val: -0.0647132970283
cost: traning 68.94899988174438 ,prediction: 23.480000019073486
alpha 10
tr: 0.41626481071,val: -0.0171265602383
cost: traning 70.18400001525879 ,prediction: 23.729000091552734
alpha 100
tr: 0.140014086698,val: -0.00554265024942
cost: traning 70.04500007629395 ,prediction: 23.651000022888184
alpha 1000
tr: 0.0434256382308,val: -0.00340556202188
cost: traning 70.57400012016296 ,prediction: 23.787999868392944
alpha 10000
tr: 0.00933880509773,val: -0.00312605877015
cost: traning 71.3529999256134 ,prediction: 23.742000102996826
alpha 100000
tr: -0.00963925528833,val: -0.00309686404652
cost: traning 71.73699998855591 ,prediction: 23.811000108718872
alpha 1000000
tr: -0.0105323696769,val: -0.00309393085464
cost: traning 72.15199995040894 ,prediction: 23.814000129699707
alpha 10000000
tr: -0.0106175512726,val: -0.00309363739683
cost: traning 73.76200008392334 ,prediction: 23.741999864578247
alpha 100000000
tr: -0.0106260318887,val: -0.00309360804967
cost: traning 74.12899994850159 ,prediction: 23.506999969482422
alpha 1000000000
tr: -0.0106268795782,val: -0.00309360511492
cost: traning 80.87199997901917 ,prediction: 23.98800015449524
alpha 10000000000
tr: -0.0106269643435,val: -0.00309360482143
cost: traning 75.84599995613098 ,prediction: 23.777999877929688
alpha 100000000000
tr: -0.0106269728199,val: -0.00309360479208
cost: traning 76.46399998664856 ,prediction: 23.931999921798706
'''