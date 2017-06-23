from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import utils
#%%
class ChunkGaussianKBest:
    def __init__(self, kf_k, seed, **kwargs):
        self.kf_k = kf_k
        self.seed = seed
        self.model_config = kwargs

    def fit(self, tr_chunk, ts_chunk):


        model_Y = tr_chunk['y'].values
        model_X = tr_chunk.drop(['id','y'], 1)
        
        
        test_Y = ts_chunk['y'].values
        test_X = ts_chunk.drop(['id','y'], 1)
        #%
        '''
         2. replace NaN with median value
        '''
        X = model_X
        self.imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
        self.imputer.fit(X)
        model_X_imputed = self.imputer.transform(X)
        
        test_X_imputed = self.imputer.transform(test_X)
        #%
        '''
         3. normalization
        '''

        self.normalization = preprocessing.StandardScaler().fit(model_X_imputed)
        model_X_norm = self.normalization.transform(model_X_imputed)
        
        test_X_norm = self.normalization.transform(test_X_imputed)
        
        
        '''
        4. kbest for over-fitting problem
        '''
        self.kbest =  SelectKBest(mutual_info_regression, k=1)
        self.kbest.fit(model_X_norm, model_Y)
        
        tr_X_kbest = self.kbest.transform(model_X_norm)
        test_X_kbest = self.kbest.transform(test_X_norm)
        print('kbest:',np.argmax(self.kbest.scores_))

        '''
        Apply regression
        '''
        
        X = tr_X_kbest 
        Y = model_Y
        
        skf = KFold(n_splits=self.kf_k,shuffle=True,random_state=self.seed)
        kf_i = 0
        self.models = []
        for tr_idx, val_idx in skf.split(X):
            print('KF',kf_i,end=',')
            tr_X, tr_Y = X[tr_idx,:], Y[tr_idx]
            val_X, val_Y = X[val_idx,:], Y[val_idx]
        
            model = GaussianProcessRegressor(alpha=self.model_config['alpha'],
                      random_state=self.model_config['random_state'])
            model.fit(tr_X, tr_Y)
            tr_Y_pred = model.predict(tr_X)
            tr_r = utils.cal_r(tr_Y,tr_Y_pred)
            print('tr:',tr_r,end=',')
            val_Y_pred = model.predict(val_X)
            val_r = utils.cal_r(val_Y,val_Y_pred)
            print('val:',val_r,end=',')
            
            test_Y_pred = model.predict(test_X_kbest)
            test_r = utils.cal_r(test_Y,test_Y_pred)
            print('test:',test_r,end='')
            
            # discard kf model whose val R is too low
            if val_r > 0:
                self.models.append(model)
                print(' (SAVED)')
            else:
                print(' (DISCARD)')
            kf_i += 1
        
        if len(self.models) != 0:
            tr_Y_sum = np.zeros(Y.shape[0])
            test_Y_sum = np.zeros(test_Y.shape[0])
            for model in self.models:
                tr_Y_sum += model.predict(X)
                test_Y_sum += model.predict(test_X_kbest)
                
            tr_Y_pred = tr_Y_sum/len(self.models)
            test_Y_pred = test_Y_sum/len(self.models)
            
            tr_r = utils.cal_r(Y,tr_Y_pred)
            print('AVERAGE tr:',tr_r,end=',')
            test_r = utils.cal_r(test_Y,test_Y_pred)
            print('test:',test_r)
        else:
            print('WARNING: no model found.')
        
        return self
    def predict(self,X):
        
        # handle chunk or extracted X 
        if 'id' in X: X = X.drop('id', 1)
        if 'y' in X: X = X.drop('y', 1)
        
        X = self.imputer.transform(X)
        X = self.normalization.transform(X)
        X = self.kbest.transform(X)
        Y_sum = np.zeros(X.shape[0])
        for model in self.models:
            Y_sum += model.predict(X)
            
        Y_pred = Y_sum/len(self.models)
        
        return Y_pred
    def __str__(self):
        return "ChunkGaussianWithKBest(kfold="+str(self.kf_k)+",seed="+str(self.seed)+')'
        