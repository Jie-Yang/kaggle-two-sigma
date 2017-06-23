
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import mutual_info_regression
from utils import cal_r


class Model1L:
    def __init__(self, seed, alpha, kbest_k):

        self.alpha = alpha
        self.seed = seed
        self.kbest_k = kbest_k


    def fit(self, tr_chunk):
        '''
         1. Remove observations whose Y are saturated values
        '''
        #tr_chunk = tr_chunk.loc[~tr_chunk['y'].isin(y_saturated_values)]
        tr_X = tr_chunk.drop(['id','y','timestamp'], 1)
        tr_Y = tr_chunk['y'].values  
        '''
         2. replace NaN with median value
         use SUPER_VALUES_selected_feature_ids to accelerate processing
        '''
        self.imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
        self.imputer.fit(tr_X)
        tr_X = self.imputer.transform(tr_X)

        '''
         3. normalization
        '''
        self.normalization = preprocessing.StandardScaler().fit(tr_X)
        tr_X= self.normalization.transform(tr_X)
        '''
        4. kbest for over-fitting problem
        '''
        self.kbest =  SelectKBest(mutual_info_regression, k=self.kbest_k)
        self.kbest.fit(tr_X, tr_Y)
        tr_X = self.kbest.transform(tr_X)
        '''
        5. Apply regression
        '''
        self.model = GaussianProcessRegressor(alpha=self.alpha,
                  random_state=self.seed)
        
        self.model.fit(tr_X, tr_Y)
        
        tr_Y_pred = self.model.predict(tr_X)
        
        return cal_r(tr_Y,tr_Y_pred)
        
    

    def predict(self,X):
        
        if 'id' in X: X = X.drop('id', 1)  
        if 'timestamp' in X: X = X.drop('timestamp', 1)  
        if 'y' in X: X = X.drop('y', 1)        
        X = self.imputer.transform(X)
        X = self.normalization.transform(X)
        X = self.kbest.transform(X)
        y_pred = self.model.predict(X)  
        
        return y_pred
        
    def val(self, val_chunk):
        
        Y = val_chunk['y'].values
        X = val_chunk.drop('y', 1)
        
        Y_pred = self.predict(X)
        
        return cal_r(Y, Y_pred)