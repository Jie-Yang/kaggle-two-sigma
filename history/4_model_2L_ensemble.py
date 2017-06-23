import time
import os
import utils
import numpy as np

#%%
test_chunk_Final = utils.read_variable('output/test_data')
test_chunk_Final_Y = test_chunk_Final['y']

#%%
chunk_i = 98 # no model generated on this chunk because of poor CV score.
print('processing chunk:',chunk_i)
test_chunk_1L = utils.read_variable('output/tr_chunks/'+str(chunk_i))
test_chunk_1L_Y = test_chunk_1L['y']
#%%
'''

Test Average Y of 4 models

each model would cost GBs memory, so can NOT load all models in memory
e.g. 4 models occupy 70% of 16GB memory
Hence, have to save intermedia results (e.g. y_pred of each 1L model) from memory into hard-drive.
'''

models_1L = []
for chunk_i in range(5):
    file_path = 'E:/two-sigma/output/chunk_gaussian_kbest/'+str(chunk_i)
    
    print('check model on chunk',chunk_i,end='...')
    t0=time.time()
    if os.path.isfile(file_path):
        model = utils.read_variable(file_path)
        print('kbest i:',np.argmax(model.kbest.scores_),end='...')
        models_1L.append(model)
        print('cost',int(time.time()-t0),'sec')
    else:
        print('No Model Found.')

        
#%%
'''
Apply every model independently on 1L testing chunk (98th chunk, which does not have corresponding model produced.)
save Y_Pred of test chunk into files
'''
for model_i in range(100):
    file_path = 'E:/two-sigma/output/chunk_gaussian_kbest/'+str(model_i)
    
    print('apply model',model_i,'on chunk 98',end='...')
    t0=time.time()
    if os.path.isfile(file_path):
        model = utils.read_variable(file_path)
        print('kbest i:',np.argmax(model.kbest.scores_),end='...')
        print('cost',int(time.time()-t0),'sec')
        
        # predict testing chunk, and save Y in file
        to = time.time()
        test_Y_pred = model.predict(test_chunk_1L)
        test_r = utils.cal_r(test_chunk_1L_Y,test_Y_pred)
        print('test:',test_r, 'cost',int(time.time()-t0),'sec')
        utils.save_variable(test_Y_pred,'E:/two-sigma/output/chunk_98_gaussian_kbest_Y_pred/'+str(model_i))
    else:
        print('No Model Found.')

#%%
'''
check model on chunk 0...kbest i: 88...cost 13 sec
test: -0.00879861627212 cost 21 sec
check model on chunk 1...kbest i: 88...cost 40 sec
test: -0.00431009568512 cost 61 sec
check model on chunk 2...kbest i: 88...cost 13 sec
test: -0.00962493789179 cost 20 sec
check model on chunk 3...kbest i: 0...cost 13 sec
test: -0.0170110256509 cost 20 sec
check model on chunk 4...No Model Found.
check model on chunk 5...kbest i: 0...cost 13 sec
test: -0.0212649323142 cost 20 sec
check model on chunk 6...kbest i: 0...cost 41 sec
test: 0.00555311056997 cost 61 sec
check model on chunk 7...kbest i: 0...cost 55 sec
test: -0.0321056317219 cost 81 sec
check model on chunk 8...kbest i: 88...cost 27 sec
test: 0.0051057916207 cost 41 sec
check model on chunk 9...kbest i: 0...cost 27 sec
test: -0.0265697143881 cost 41 sec
check model on chunk 10...kbest i: 0...cost 26 sec
test: -0.0161352463253 cost 40 sec
check model on chunk 11...kbest i: 0...cost 26 sec
test: -0.0217091439107 cost 40 sec
check model on chunk 12...kbest i: 100...cost 13 sec
test: -33.5672142562 cost 20 sec
check model on chunk 13...kbest i: 0...cost 13 sec
test: -0.0257988561033 cost 20 sec
check model on chunk 14...kbest i: 0...cost 54 sec
test: -0.0556796197101 cost 80 sec
check model on chunk 15...kbest i: 88...cost 13 sec
test: -0.014897455215 cost 20 sec
check model on chunk 16...kbest i: 0...cost 13 sec
test: -0.0215643337532 cost 19 sec
check model on chunk 17...kbest i: 88...cost 13 sec
test: -0.00654986179357 cost 20 sec
check model on chunk 18...kbest i: 88...cost 40 sec
test: -0.00523989754714 cost 61 sec
check model on chunk 19...kbest i: 88...cost 13 sec
test: -0.00463764535361 cost 20 sec
check model on chunk 20...kbest i: 88...cost 13 sec
test: -0.0101535335683 cost 20 sec
check model on chunk 21...kbest i: 0...cost 27 sec
test: -0.0217209423379 cost 41 sec
check model on chunk 22...kbest i: 0...cost 13 sec
test: -0.0365876817369 cost 21 sec
check model on chunk 23...kbest i: 0...cost 14 sec
test: 0.0189061276802 cost 21 sec
check model on chunk 24...No Model Found.
check model on chunk 25...kbest i: 0...cost 13 sec
test: -0.0260088311964 cost 20 sec
check model on chunk 26...kbest i: 88...cost 13 sec
test: 0.010021876595 cost 20 sec
check model on chunk 27...kbest i: 0...cost 27 sec
test: -0.0150156162484 cost 41 sec
check model on chunk 28...kbest i: 88...cost 27 sec
test: -0.0137616137522 cost 41 sec
check model on chunk 29...kbest i: 0...cost 27 sec
test: -0.0292518854279 cost 40 sec
check model on chunk 30...kbest i: 0...cost 27 sec
test: -0.00496617427452 cost 40 sec
check model on chunk 31...kbest i: 88...cost 41 sec
test: -0.0101449326241 cost 61 sec
check model on chunk 32...kbest i: 88...cost 13 sec
test: -0.00814389811427 cost 20 sec
check model on chunk 33...kbest i: 0...cost 13 sec
test: -0.0292433146262 cost 20 sec
check model on chunk 34...kbest i: 99...cost 13 sec
test: -3.17369656678 cost 20 sec
check model on chunk 35...kbest i: 88...cost 26 sec
test: -0.0137474669643 cost 40 sec
check model on chunk 36...kbest i: 88...cost 13 sec
test: -0.0213212473829 cost 20 sec
check model on chunk 37...kbest i: 88...cost 13 sec
test: 0.00763318234728 cost 20 sec
check model on chunk 38...kbest i: 88...cost 13 sec
test: -0.00290212617647 cost 20 sec
check model on chunk 39...kbest i: 88...cost 43 sec
test: -0.0237705908129 cost 65 sec
check model on chunk 40...kbest i: 88...cost 13 sec
test: -0.0222596858499 cost 20 sec
check model on chunk 41...kbest i: 0...cost 13 sec
test: -0.0342569357428 cost 20 sec
check model on chunk 42...kbest i: 0...cost 13 sec
test: -0.0169688199987 cost 20 sec
check model on chunk 43...kbest i: 0...cost 13 sec
test: -0.00582289565182 cost 20 sec
check model on chunk 44...kbest i: 88...cost 55 sec
test: -0.0234357327328 cost 82 sec
check model on chunk 45...kbest i: 88...cost 27 sec
test: -0.00335253127236 cost 41 sec
check model on chunk 46...kbest i: 0...cost 13 sec
test: -0.0283100628152 cost 20 sec
check model on chunk 47...kbest i: 91...cost 14 sec
test: -16.0214055815 cost 20 sec
check model on chunk 48...kbest i: 0...cost 13 sec
test: -0.0279847715177 cost 20 sec
check model on chunk 49...kbest i: 99...cost 13 sec
test: -34.0135657422 cost 20 sec
check model on chunk 50...kbest i: 88...cost 13 sec
test: -0.0122761056205 cost 20 sec
check model on chunk 51...kbest i: 0...cost 13 sec
test: -0.018358289604 cost 20 sec
check model on chunk 52...No Model Found.
check model on chunk 53...No Model Found.
check model on chunk 54...kbest i: 0...cost 13 sec
test: -0.0347916397523 cost 20 sec
check model on chunk 55...kbest i: 88...cost 13 sec
test: -0.016023820476 cost 20 sec
check model on chunk 56...No Model Found.
check model on chunk 57...No Model Found.
check model on chunk 58...kbest i: 88...cost 13 sec
test: -0.00869979225943 cost 20 sec
check model on chunk 59...kbest i: 88...cost 41 sec
test: -0.0111831279794 cost 61 sec
check model on chunk 60...kbest i: 100...cost 14 sec
test: -101.184186261 cost 20 sec
check model on chunk 61...kbest i: 0...cost 13 sec
test: -0.0215725559885 cost 20 sec
check model on chunk 62...No Model Found.
check model on chunk 63...kbest i: 0...cost 13 sec
test: -0.0259731835335 cost 20 sec
check model on chunk 64...kbest i: 0...cost 41 sec
test: -0.028267705989 cost 61 sec
check model on chunk 65...kbest i: 88...cost 14 sec
test: -0.0117500900662 cost 21 sec
check model on chunk 66...kbest i: 0...cost 27 sec
test: -0.0153469093448 cost 41 sec
check model on chunk 67...No Model Found.
check model on chunk 68...kbest i: 88...cost 42 sec
test: -0.0265026288132 cost 63 sec
check model on chunk 69...kbest i: 0...cost 28 sec
test: -0.00614945406934 cost 41 sec
check model on chunk 70...kbest i: 88...cost 14 sec
test: -0.00842299316253 cost 21 sec
check model on chunk 71...kbest i: 0...cost 14 sec
test: -0.0243914660498 cost 20 sec
check model on chunk 72...kbest i: 99...cost 14 sec
test: -4.52934066365 cost 22 sec
check model on chunk 73...kbest i: 0...cost 28 sec
test: -0.0435627480884 cost 42 sec
check model on chunk 74...kbest i: 88...cost 14 sec
test: -0.00112734354885 cost 21 sec
check model on chunk 75...kbest i: 0...cost 14 sec
test: -0.0180959086816 cost 20 sec
check model on chunk 76...kbest i: 0...cost 28 sec
test: -0.0324090892718 cost 41 sec
check model on chunk 77...kbest i: 88...cost 14 sec
test: -0.00542210249504 cost 21 sec
check model on chunk 78...kbest i: 99...cost 14 sec
test: -78.7910222101 cost 20 sec
check model on chunk 79...kbest i: 0...cost 28 sec
test: -0.0333493692704 cost 41 sec
check model on chunk 80...kbest i: 88...cost 14 sec
test: -0.0105655871103 cost 21 sec
check model on chunk 81...kbest i: 0...cost 28 sec
test: -0.0182043858147 cost 42 sec
check model on chunk 82...kbest i: 0...cost 14 sec
test: -0.0126109371895 cost 21 sec
check model on chunk 83...kbest i: 0...cost 14 sec
test: 0.0147506270628 cost 21 sec
check model on chunk 84...kbest i: 88...cost 43 sec
test: 0.00963331407968 cost 64 sec
check model on chunk 85...kbest i: 88...cost 15 sec
test: -0.0143873288569 cost 22 sec
check model on chunk 86...kbest i: 88...cost 28 sec
test: 0.00766682486131 cost 42 sec
check model on chunk 87...No Model Found.
check model on chunk 88...No Model Found.
check model on chunk 89...kbest i: 88...cost 14 sec
test: -0.00993117529758 cost 21 sec
check model on chunk 90...kbest i: 0...cost 29 sec
test: -0.0259618557003 cost 42 sec
check model on chunk 91...kbest i: 99...cost 14 sec
test: -15.7105012924 cost 21 sec
check model on chunk 92...No Model Found.
check model on chunk 93...kbest i: 0...cost 14 sec
test: 0.0129056214987 cost 21 sec
check model on chunk 94...No Model Found.
check model on chunk 95...kbest i: 88...cost 29 sec
test: -0.0127166718404 cost 44 sec
check model on chunk 96...No Model Found.
check model on chunk 97...kbest i: 88...cost 15 sec
test: -0.0126667815664 cost 21 sec
check model on chunk 98...No Model Found.
check model on chunk 99...kbest i: 88...cost 44 sec
test: -0.0219722420053 cost 66 sec
'''        
#%%
'''
sum all model Ys, and cal the average value as the final output

'''
model_c = 0
y_pred_sum = np.zeros(len(test_chunk_1L_Y))
model_r = []
good_models_i = []

for model_i in range(100):
    file_path = 'E:/two-sigma/output/chunk_98_gaussian_kbest_Y_pred/'+str(model_i)
    
    t0=time.time()
    if os.path.isfile(file_path):
        y_pred = utils.read_variable(file_path)
        temp_r = utils.cal_r(test_chunk_1L_Y,y_pred)
        print('model r:',temp_r)
        if temp_r >0:
            y_pred_sum += y_pred
            model_c += 1
            model_r.append(temp_r)
            good_models_i.append(model_i)
            
    else:
        print(model_i,'No Model Found.')
        
y_pred_mean = y_pred_sum / model_c
print('R of average 1L testing Y:',utils.cal_r(test_chunk_1L_Y,y_pred_mean))
print('selected 1L models:',good_models_i)
#%%
from shutil import copyfile

for model_i in good_models_i:
    t0=time.time()
    print('copying good model',model_i, end='...')
    file_path_src = 'E:/two-sigma/output/chunk_gaussian_kbest/'+str(model_i)
    file_path_dst = 'E:/two-sigma/output/1L_good_models/'+str(model_i)
    copyfile(file_path_src,file_path_dst)
    print('cost',int(time.time()-t0),'sec')


#%%
'''
no filter on temp_r, 86 models is remained from 86
R of average testing Y: -1.91344393058

filter on temp_r >0, 9 models is remained from 86
R of average testing Y: 0.0216042683604
'''

#%%
'''
validate ensemble model on Final test chunk
'''
y_pred_sum = np.zeros(len(test_chunk_Final_Y))
for model_i in good_models_i:
    file_path = 'E:/two-sigma/output/chunk_gaussian_kbest_test_data_Y_pred/'+str(model_i)
    
    y_pred = utils.read_variable(file_path)
    temp_r = utils.cal_r(test_chunk_Final_Y,y_pred)
    print('model r:',temp_r)
    y_pred_sum += y_pred

y_pred_mean = y_pred_sum / len(good_models_i)
print('R of average testing Y:',utils.cal_r(test_chunk_Final_Y,y_pred_mean))

#%%
'''
R of average testing Y: 0.013725691326
'''