import numpy as np
import pandas as pd
import utils
from ChunkCVR import ChunkCVR
from ChunkTree import ChunkTree

#%%

chunk_i = 45
tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))

test_chunk = utils.read_variable('output/test_data')

tr_Y = tr_chunk['y']

#%%
C = 1
kernel='sigmoid'
gamma='auto'
tol=1e-3

model = ChunkCVR(kf_k=5, seed=13, kernel=kernel,C=C,gamma=gamma,tol=tol)
model.fit(tr_chunk, test_chunk)

#%%
'''
Result: no matter how to change parameters like C, kernel, gamma, and tol, there is no any improvement

C = 1.0
kernel='rbf'
gamma='auto'
tol=1e-3

KF 0,tr: -0.206720921687,val: -0.199090411886,test: -0.171216439608 (SAVED)
KF 1,tr: -0.222389298715,val: -0.208357570655,test: -0.184041916324 (SAVED)
KF 2,tr: -0.203784348487,val: -0.211136156614,test: -0.171216439608 (SAVED)
KF 3,tr: -0.195383679304,val: -0.211565672849,test: -0.16532344652 (SAVED)
KF 4,tr: -0.205528411936,val: -0.203959448218,test: -0.171216439608 (SAVED)
AVERAGE tr: -0.206745361882,test: -0.172602936305
'''