import numpy as np
import pandas as pd
import utils
from ChunkLinearRegression import ChunkLinearRegression
from ChunkTree import ChunkTree

#%%

chunk_i = 45
tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))

test_chunk = utils.read_variable('output/test_data')

tr_Y = tr_chunk['y']
#%%
'''
Linear Regression
'''


model_linear = ChunkLinearRegression(kf_k=5, seed=13)
model_linear.fit(tr_chunk, test_chunk)

tr_Y_pred = model_linear.predict(tr_chunk)

print('r:',utils.cal_r(tr_chunk['y'],tr_Y_pred))

'''
KF 0,tr: 0.136013944038,val: -607.864299346,test: -51.432700711 (DISCARD)
KF 1,tr: 0.126612607023,val: -42707538498.9,test: -245495431.615 (DISCARD)
KF 2,tr: 0.140904353882,val: -0.37319746263,test: -9.83011647629 (SAVED)
KF 3,tr: 0.132568523411,val: -0.625998947552,test: -43.6620637906 (SAVED)
KF 4,tr: 0.126012579869,val: -0.266122719199,test: -48.915686375 (SAVED)
AVERAGE tr: 0.0448775523981,test: -27.5904174498
r: 0.0448775523981
'''
#%%
'''
Tree Regression
'''

model = ChunkTree(kf_k=5, seed=13, max_depth=5)
model.fit(tr_chunk, test_chunk)

'''
KF 0,tr: 0.167128264288,val: -0.173966241986,test: -0.16713462088 (SAVED)
KF 1,tr: 0.141213509881,val: -0.141064110418,test: -0.11927849284 (SAVED)
KF 2,tr: 0.183043827304,val: -0.199940206966,test: -0.18639851534 (SAVED)
KF 3,tr: 0.179065693846,val: -0.240717113104,test: -0.220164110866 (SAVED)
KF 4,tr: 0.114158458319,val: -0.134298683888,test: -0.121935408889 (SAVED)
AVERAGE tr: 0.178408809789,test: -0.0999002162407
'''
#%%
for max_depth in range(1,100,1):
    print('##############',max_depth,'##############')
    model = ChunkTree(kf_k=5, seed=13, max_depth=max_depth)
    model.fit(tr_chunk, test_chunk)
    
'''
output:
    1.best test R achieve when max_depth is 1, but still smaller than 0
    2. test R decrease when max_depth increase
    
############## 1 ##############
AVERAGE tr: 0.0552347211041,test: -0.0148390117315
############## 2 ##############
AVERAGE tr: 0.0867527336794,test: -0.0413208351181
############## 3 ##############
AVERAGE tr: 0.117764099285,test: -0.066812288417
############## 4 ##############
AVERAGE tr: 0.146866351176,test: -0.080271442188
############## 5 ##############
AVERAGE tr: 0.178933785791,test: -0.0956814529269
'''
#%%
import matplotlib.pyplot as plt

#plt.hist(tr_Y_pred, bins=100)  # plt.hist passes it's arguments to np.histogram
#plt.title("Distribution of Y")
#plt.show()

plt.plot(tr_Y, tr_Y_pred, 'rs')
plt.axis([-0.1, 0.15, -0.1, 0.15])
plt.xlabel('true')
plt.ylabel('pred')
plt.show()


#%%
'''
Bayesian
'''
from ChunkBayesian import ChunkBayesian

tol  = 1.e-3 # no significant improvement by changing this value
alpha_1 = 1.e-6 # no significant improvement
alpha_2 = 1.e-6 # no significant improvement
lambda_1 = 1.e-6 # no significant improvement
lambda_2 = 1.e-6 # no significant improvement

lambda_2 = 1.e-10
for i in range(12):
    model = ChunkBayesian(kf_k=5, seed=13, 
                          tol=tol,
                          alpha_1=alpha_1,
                          alpha_2=alpha_2,
                          lambda_1=lambda_1,
                          lambda_2=lambda_2)
    model.fit(tr_chunk, test_chunk)
    lambda_2 *= 10

#%%
'''

NON-LINEAR MODEL

'''
#%%
'''
Gaussian process regression 

ref to file 2_model_survey_gaussian

Result: almost perfect modelling on training dataset (r:0.99), but really bad result on val and testing.
Tried to solve this over-fitting problem by adjusting alpha, but the best val R value is still below zero.
'''

