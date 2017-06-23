import pickle
from sklearn.metrics import r2_score
import numpy as np
    
def save_variable(var,file_name):
    pkl_file = open(file_name, 'wb')
    pickle.dump(var, pkl_file, -1)
    pkl_file.close()
    
def read_variable(file_name):
    pkl_file = open(file_name, 'rb')
    var = pickle.load(pkl_file)
    pkl_file.close()
    return var
    
def cal_r(y_true, y_pred):
    #https://www.kaggle.com/c/two-sigma-financial-modeling/details/evaluation
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2)*np.sqrt(np.absolute(r2))
    # r in kaggle platform will be clipped at -1
    return r;