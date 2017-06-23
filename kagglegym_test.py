import kagglegym
import numpy as np
import random
import time as t
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import mutual_info_regression

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()




#%%
'''
predict target
'''



processed_test_c = 0
while True:
    target = observation.target
    features = observation.features

    target.y = np.zeros(len(target.y))
    
    processed_test_c += len(target.y)

    observation, reward, done, info = env.step(target)
    #print(processed_test_c,'reward:',reward)
    
    times =  features.timestamp.unique()
    print(times)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break
    
