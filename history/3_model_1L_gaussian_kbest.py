import numpy as np
import utils

from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import time
from sklearn.model_selection import KFold
from ChunkGaussianKBest import ChunkGaussianKBest
import os
#%%
'''
use 5 CV to validate model, and only keep models whose val R score > 0.
1. no model is generated on some tr chunks whose R is alway < 0
'''

for chunk_i in range(100):
    print('processing chunk:',chunk_i)
    
    file_path = 'E:/two-sigma/output/chunk_gaussian_kbest/'+str(chunk_i)
    
    if os.path.isfile(file_path):
        print('model exist')
    else:
        tr_chunk = utils.read_variable('E:/two-sigma/output/tr_chunks/'+str(chunk_i))
        test_chunk = utils.read_variable('E:/two-sigma/output/test_data')
        tr_Y = tr_chunk['y']
    
        for seed in [13,17,97,23,111]: 
            model = ChunkGaussianKBest(kf_k=5, seed=seed,alpha=1e-10, random_state=seed)
            model.fit(tr_chunk, test_chunk)
            if len(model.models)>0:
                break
        
        utils.save_variable(model,file_path)
        

#%%
model_c = 0
for chunk_i in range(100):
    file_path = 'E:/two-sigma/output/chunk_gaussian_kbest/'+str(chunk_i)
    
    print('check chunk',chunk_i,end='...')
    t0=time.time()
    if os.path.isfile(file_path):
        model_c +=1
        model = utils.read_variable(file_path)
        print('kbest i:',np.argmax(model.kbest.scores_),end='...')
    print('cost',int(time.time()-t0),'sec')
        
print(model_c,'of 100 generated.')
#%%
'''
Results: 
    1. gaussian with kbest produce models whose val score>0 on 86 of 100 chunks
    2. index of feature is selected by KBest are: 0, 88, 91, 99, 100.
    Opportunity: run another round of modelling on these five features. 
    
Note: following loading speed (model size is from 1 to 3 Gb.) is based on USB2, if use USB3, the speed could 2 or 3 times faster.
check chunk 0...kbest i: 88...cost 1 sec
check chunk 1...kbest i: 88...cost 74 sec
check chunk 2...kbest i: 88...cost 36 sec
check chunk 3...kbest i: 0...cost 37 sec
check chunk 4...cost 0 sec
check chunk 5...kbest i: 0...cost 37 sec
check chunk 6...kbest i: 0...cost 111 sec
check chunk 7...kbest i: 0...cost 147 sec
check chunk 8...kbest i: 88...cost 72 sec
check chunk 9...kbest i: 0...cost 74 sec
check chunk 10...kbest i: 0...cost 74 sec
check chunk 11...kbest i: 0...cost 74 sec
check chunk 12...kbest i: 100...cost 36 sec
check chunk 13...kbest i: 0...cost 36 sec
check chunk 14...kbest i: 0...cost 148 sec
check chunk 15...kbest i: 88...cost 36 sec
check chunk 16...kbest i: 0...cost 36 sec
check chunk 17...kbest i: 88...cost 36 sec
check chunk 18...kbest i: 88...cost 109 sec
check chunk 19...kbest i: 88...cost 36 sec
check chunk 20...kbest i: 88...cost 36 sec
check chunk 21...kbest i: 0...cost 73 sec
check chunk 22...kbest i: 0...cost 37 sec
check chunk 23...kbest i: 0...cost 36 sec
check chunk 24...cost 0 sec
check chunk 25...kbest i: 0...cost 37 sec
check chunk 26...kbest i: 88...cost 36 sec
check chunk 27...kbest i: 0...cost 74 sec
check chunk 28...kbest i: 88...cost 73 sec
check chunk 29...kbest i: 0...cost 74 sec
check chunk 30...kbest i: 0...cost 74 sec
check chunk 31...kbest i: 88...cost 109 sec
check chunk 32...kbest i: 88...cost 36 sec
check chunk 33...kbest i: 0...cost 37 sec
check chunk 34...kbest i: 99...cost 36 sec
check chunk 35...kbest i: 88...cost 73 sec
check chunk 36...kbest i: 88...cost 36 sec
check chunk 37...kbest i: 88...cost 36 sec
check chunk 38...kbest i: 88...cost 36 sec
check chunk 39...kbest i: 88...cost 109 sec
check chunk 40...kbest i: 88...cost 36 sec
check chunk 41...kbest i: 0...cost 37 sec
check chunk 42...kbest i: 0...cost 37 sec
check chunk 43...kbest i: 0...cost 37 sec
check chunk 44...kbest i: 88...cost 145 sec
check chunk 45...kbest i: 88...cost 73 sec
check chunk 46...kbest i: 0...cost 37 sec
check chunk 47...kbest i: 91...cost 36 sec
check chunk 48...kbest i: 0...cost 37 sec
check chunk 49...kbest i: 99...cost 36 sec
check chunk 50...kbest i: 88...cost 36 sec
check chunk 51...kbest i: 0...cost 37 sec
check chunk 52...cost 0 sec
check chunk 53...cost 0 sec
check chunk 54...kbest i: 0...cost 37 sec
check chunk 55...kbest i: 88...cost 36 sec
check chunk 56...cost 0 sec
check chunk 57...cost 0 sec
check chunk 58...kbest i: 88...cost 36 sec
check chunk 59...kbest i: 88...cost 109 sec
check chunk 60...kbest i: 100...cost 36 sec
check chunk 61...kbest i: 0...cost 36 sec
check chunk 62...cost 0 sec
check chunk 63...kbest i: 0...cost 37 sec
check chunk 64...kbest i: 0...cost 111 sec
check chunk 65...kbest i: 88...cost 36 sec
check chunk 66...kbest i: 0...cost 74 sec
check chunk 67...cost 0 sec
check chunk 68...kbest i: 88...cost 109 sec
check chunk 69...kbest i: 0...cost 74 sec
check chunk 70...kbest i: 88...cost 36 sec
check chunk 71...kbest i: 0...cost 36 sec
check chunk 72...kbest i: 99...cost 36 sec
check chunk 73...kbest i: 0...cost 74 sec
check chunk 74...kbest i: 88...cost 36 sec
check chunk 75...kbest i: 0...cost 37 sec
check chunk 76...kbest i: 0...cost 74 sec
check chunk 77...kbest i: 88...cost 36 sec
check chunk 78...kbest i: 99...cost 36 sec
check chunk 79...kbest i: 0...cost 74 sec
check chunk 80...kbest i: 88...cost 36 sec
check chunk 81...kbest i: 0...cost 74 sec
check chunk 82...kbest i: 0...cost 37 sec
check chunk 83...kbest i: 0...cost 37 sec
check chunk 84...kbest i: 88...cost 109 sec
check chunk 85...kbest i: 88...cost 36 sec
check chunk 86...kbest i: 88...cost 72 sec
check chunk 87...cost 0 sec
check chunk 88...cost 0 sec
check chunk 89...kbest i: 88...cost 36 sec
check chunk 90...kbest i: 0...cost 73 sec
check chunk 91...kbest i: 99...cost 36 sec
check chunk 92...cost 0 sec
check chunk 93...kbest i: 0...cost 36 sec
check chunk 94...cost 0 sec
check chunk 95...kbest i: 88...cost 73 sec
check chunk 96...cost 0 sec
check chunk 97...kbest i: 88...cost 36 sec
check chunk 98...cost 0 sec
check chunk 99...kbest i: 88...cost 109 sec
86 of 100 generated.
'''