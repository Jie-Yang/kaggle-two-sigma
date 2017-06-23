import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
original_data = pd.read_hdf('data/train.h5')
original_data.info()



#%%
'''
Check stationarity of Y from a certain obj ID
'''
id_timestamp_y = original_data[['id','timestamp','y']]

#%%
print('unique obj ids:',len(np.unique(id_timestamp_y.id)))
for obj_id in np.unique(id_timestamp_y.id):
    print(obj_id,id_timestamp_y[id_timestamp_y.id==obj_id].shape[0])
#%%
obj_id = 1860

timestamp_y = id_timestamp_y[id_timestamp_y.id==obj_id]
plt.xlabel('Time')
plt.ylabel('Y')
#plt.errorbar(ts_timestamp, ts_mean, yerr=ts_std)
plt.plot(timestamp_y.timestamp, timestamp_y.y)
plt.title("Y of obj "+str(obj_id))
plt.xlim([min(timestamp_y.timestamp),max(timestamp_y.timestamp)])
plt.show()

#%%
'''
check stats of Y (from all objs) at different timestamp
'''

id_timestamp_y = original_data[['id','timestamp','y']]
groups_timestamp = id_timestamp_y.groupby('timestamp')

ts_n = len(groups_timestamp.groups.keys())
ts_timestamp = np.zeros(ts_n)
ts_mean = np.zeros(ts_n)
ts_std = np.zeros(ts_n)
ts_i = 0

for key in groups_timestamp.groups.keys():
    print('Timestamp:',key,end=',')
    row_ids = groups_timestamp.groups[key]
    gp_data = id_timestamp_y.ix[row_ids]
    ts_mean[ts_i] = gp_data['y'].mean()
    ts_std[ts_i] = gp_data['y'].std()
    ts_timestamp[ts_i] = key
    print('SIZE:',len(row_ids))
    ts_i +=1

#%%
plt.figure()
plt.xlabel('Time')
plt.ylabel('Y')
#plt.errorbar(ts_timestamp, ts_mean, yerr=ts_std)
plt.plot(ts_timestamp, ts_mean,label='$\mu$')
plt.plot(ts_timestamp, ts_mean+ts_std,label='$\mu+\sigma$')
plt.plot(ts_timestamp, ts_mean-ts_std,label='$\mu-\sigma$')

#obj_id = 1860
#timestamp_y = id_timestamp_y[id_timestamp_y.id==obj_id]
#plt.plot(timestamp_y.timestamp, timestamp_y.y, label='Obj'+str(obj_id))

plt.xlim(min(ts_timestamp), max(ts_timestamp))
plt.title('$\mu\pm\sigma$')
plt.legend()
plt.show()

#%%
'''
Clustering on (Mean Y, Timestamp)

'''

clustering_X = pd.DataFrame({'timestamp':ts_timestamp,'y':ts_mean})

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

    
n_clusters = []
n_sil= []
for k in range(2,30,1):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(clustering_X)
    labels = kmeans.labels_
    
    sil = silhouette_score(clustering_X, labels, metric='euclidean')
    
    print(k, sil)
    n_sil.append(sil)
    n_clusters.append(k)
    
    if sil == 1:
        break
#%%
import matplotlib.pyplot as plt

plt.figure(2)
plt.plot(n_clusters,n_sil)
plt.xlabel('k')
plt.ylabel('silhouette')

#%%
k = 3

kmeans = KMeans(n_clusters=k, random_state=0).fit(clustering_X)
labels = kmeans.labels_

clustering_X_Label = clustering_X
clustering_X_Label['label'] = labels

plt.figure(2)

for label in np.unique(labels):
    cluster = clustering_X_Label[clustering_X_Label.label==label]
    plt.plot(cluster.timestamp,cluster.y)
plt.xlabel('k')
plt.ylabel('silhouette')

