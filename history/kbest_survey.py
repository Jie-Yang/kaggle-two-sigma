
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import utils
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
#%%
kbest_i = [0, 88, 91, 99, 100]
for chunk_i in range(100):
    print(chunk_i,end='-->')
    chunk = utils.read_variable('output/tr_chunks/'+str(chunk_i))
    X = chunk.drop(['id','y'], 1).as_matrix()
    X = X[:,kbest_i]
    Y = chunk['y'].values

    X_imputed = Imputer(missing_values='NaN', strategy='median', axis=0).fit_transform(X)
    X_norm = preprocessing.StandardScaler().fit_transform(X_imputed)

    kf_i = 0
    skf = KFold(n_splits=5,shuffle=True,random_state=13)
    for tr_idx, val_idx in skf.split(X_norm):
        print('kf',kf_i,end='-->')
        kf_i += 1
        tr_X, tr_Y = X_norm[tr_idx,:], Y[tr_idx]
        val_X, val_Y = X_norm[val_idx,:], Y[val_idx]

        model = LinearRegression()
        model.fit(tr_X, tr_Y)
        tr_Y_pred = model.predict(tr_X)
        tr_r = utils.cal_r(tr_Y,tr_Y_pred)
        print('tr:',tr_r,end=',')
        val_Y_pred = model.predict(val_X)
        val_r = utils.cal_r(val_Y,val_Y_pred)
        print('val:',val_r)
        
