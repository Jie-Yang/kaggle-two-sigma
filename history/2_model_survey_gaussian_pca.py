import numpy as np
import pandas as pd
import utils
from ChunkTree import ChunkTree

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import time
from sklearn.model_selection import KFold

#%%

chunk_i = 45
tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))

test_chunk = utils.read_variable('output/test_data')

tr_Y = tr_chunk['y']


#%%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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
    
    for n_components in range(1,100,2):
        
        print('n_components:',n_components,end='-->')
    
        pca = PCA(n_components=n_components)
        pca.fit(tr_X_norm)
        
        tr_X_pca = pca.transform(tr_X_norm)
        val_X_pca =  pca.transform(val_X_norm)
        
        model = GaussianProcessRegressor(alpha=1e-10,random_state=13)
        model.fit(tr_X_pca, tr_Y)
        tr_Y_pred = model.predict(tr_X_pca)
        tr_r = utils.cal_r(tr_Y,tr_Y_pred)
        print('tr:',tr_r,end=', ')
        
        val_Y_pred = model.predict(val_X_pca)
        val_r = utils.cal_r(val_Y,val_Y_pred)
        print('val:',val_r)
    
    break
#%%
'''
Result: PCA not solve over-fitting problem

PCA
n_components: 1-->tr: 0.0870017651238, test: -98.0679086509
n_components: 3-->tr: 0.546443154653, test: -308.525575086
n_components: 5-->tr: 0.984719164772, test: -355.849904876
n_components: 7-->tr: 0.999992365307, test: -17.5517010962
n_components: 9-->tr: 0.99999238285, test: -5.98769701532
n_components: 11-->tr: 0.999992382853, test: -2.9713507691
n_components: 13-->tr: 0.999992382853, test: -2.20648807019
n_components: 15-->tr: 0.999992382853, test: -1.4833746133
n_components: 17-->tr: 0.999992382853, test: -1.35901237941
n_components: 19-->tr: 0.999992382853, test: -1.19570079994
n_components: 21-->tr: 0.999992382853, test: -1.01013447642
n_components: 23-->tr: 0.999992382853, test: -0.839910714473
... ....
n_components: 93-->tr: 0.999992382853, test: -0.124638692073
n_components: 95-->tr: 0.999992382853, test: -0.124570235039
n_components: 97-->tr: 0.999992382853, test: -0.124546596749
n_components: 99-->tr: 0.999992382853, test: -0.124519837072
'''
#%%
import matplotlib.pyplot as plt

cumsum = np.cumsum(pca.explained_variance_ratio_)

xs = range(100)
plt.plot(range(len(cumsum)), cumsum, 'rs')
plt.show()