import numpy as np
import utils

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import time
from sklearn.model_selection import KFold
from ChunkGaussianKBest import ChunkGaussianKBest

#%%
'''
Observation: Kbest with mutual_info_regression could solve over-fitting problem of gaussian process
So it is used to generated 1L models.
'''
#%%

chunk_i = 45
print('processing chunk:',chunk_i)
tr_chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))

test_chunk = utils.read_variable('output/test_data')

tr_Y = tr_chunk['y']

model = ChunkGaussianKBest(kf_k=5, seed=13,alpha=1e-10, random_state=13)
model.fit(tr_chunk, test_chunk)
    


#%%
'''
processing chunk: 0
kbest: 88
KF 0,tr: 0.0132627185327,val: -0.0243398320384,test: -0.00338971594141 (DISCARD)
KF 1,tr: 0.00982018438149,val: -0.0160816320486,test: 0.0138005866055 (DISCARD)
KF 2,tr: 0.00307570289739,val: 0.00913977859153,test: 0.00560680201576 (SAVED)
KF 3,tr: 0.00610072488182,val: -0.0323927402247,test: -0.0154713553836 (DISCARD)
KF 4,tr: 0.0159781002194,val: -0.0361629262381,test: 0.0110115264546 (DISCARD)
AVERAGE tr: 0.00588665594919,test: 0.00560680201576
processing chunk: 1
kbest: 88
KF 0,tr: 0.0150200891603,val: 0.0218895528891,test: 0.0164926972463 (SAVED)
KF 1,tr: 0.011880203847,val: 0.0244835022911,test: 0.0132552810219 (SAVED)
KF 2,tr: 0.00982613857259,val: 0.0234624959332,test: 0.0143947499918 (SAVED)
KF 3,tr: 0.0192389192283,val: -0.0255942173268,test: 0.011570294508 (DISCARD)
KF 4,tr: 0.0305448653926,val: -0.0699098647691,test: 0.0116670700097 (DISCARD)
AVERAGE tr: 0.0160457955785,test: 0.0152173563837

'''
#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor

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
    
    #f_regression, mutual_info_regression
    kbest = SelectKBest(mutual_info_regression, k=1)
    
    t0 = time.time()
    kbest.fit(tr_X_norm, tr_Y)

    
    t1 = time.time()
    tr_X_pca = kbest.transform(tr_X_norm)
    t2 = time.time()
    val_X_pca =  kbest.transform(val_X_norm)
    
    
    t3 = time.time()
    model = GaussianProcessRegressor(alpha=1e-10,random_state=13)
    model.fit(tr_X_pca, tr_Y)
    t4 = time.time()
    tr_Y_pred = model.predict(tr_X_pca)
    t5 = time.time()
    tr_r = utils.cal_r(tr_Y,tr_Y_pred)
    print('tr:',tr_r,end=', ')
    
    val_Y_pred = model.predict(val_X_pca)
    val_r = utils.cal_r(val_Y,val_Y_pred)
    print('val:',val_r, end='-->')
    print('kbest i',np.argmax(kbest.scores_),end='--')
    print('kbest fit',int(t1-t0),',gaussian fit',int(t4-t3),',pre',int(t5-t4))
    


#%%

'''
KBest with f_regression
kbest: 1-->tr: 0.0519983902672, val: -0.0794699443779-->kbest fit 0 ,gaussian fit 44 ,pre 10
kbest: 3-->tr: 0.297506861563, val: -182.640259642-->kbest fit 0 ,gaussian fit 46 ,pre 5
kbest: 5-->tr: 0.618488963699, val: -509.108685046-->kbest fit 0 ,gaussian fit 51 ,pre 5
kbest: 7-->tr: 0.842228819852, val: -523.806505835-->kbest fit 0 ,gaussian fit 47 ,pre 6
kbest: 9-->tr: 0.863696552816, val: -518.854444908-->kbest fit 0 ,gaussian fit 49 ,pre 6

KBest with mutual_info_regression
kbest: 1-->tr: 0.0184656024035, val: 0.0188379253411-->kbest fit 132 ,gaussian fit 44 ,pre 25
kbest: 3-->tr: 0.230210449096, val: -74.7083126521-->kbest fit 133 ,gaussian fit 45 ,pre 5
kbest: 5-->tr: 0.800770252156, val: -325.021800713-->kbest fit 133 ,gaussian fit 48 ,pre 6
'''

model.models