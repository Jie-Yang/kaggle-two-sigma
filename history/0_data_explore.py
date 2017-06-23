import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
original_data = pd.read_hdf('data/train.h5')
original_data.info()

print('cols:',original_data.columns.size)
#%%
def explore_col(col_name):
    original_col = original_data[col_name]
    
    print('count',len(original_col))
    print('min',np.min(original_col))
    print('max',np.max(original_col))
    print('mean',np.mean(original_col))
    print('std',np.std(original_col))
    print('nan',np.sum(np.isnan(original_col)))

    plt.hist(original_col, bins=100)  # plt.hist passes it's arguments to np.histogram
    plt.title("Distribution of "+col_name)
    plt.show()
    
    return original_col
#%%
'''
explore Y
'''
data_y = explore_col('y')

#%%
'''
considering the fact that the abnomally high numbers of samples on the left and right edge of hist plot,
   maybe it is a sign of: y value is trimmed because the real value is exceed the max/min value of the measurement.
   Hence, maybe it is an option to rule all of these values in the following modelling?
'''
y_min = np.min(data_y)
y_max = np.max(data_y)

data_y_min_count = data_y==y_min
print('sample with min Y:',sum(data_y_min_count),sum(data_y_min_count)/len(data_y))
data_y_max_count = data_y==y_max
print('sample with max Y:',sum(data_y_max_count),sum(data_y_max_count)/len(data_y))

'''
output
sample with min Y: 11173 0.0065310307256
sample with max Y: 11286 0.00659708339471

Decision: all of these samples will be removed from training dataset, but proportional kept in testing dataset.
'''

data_y_desaturated = data_y.loc[~data_y.isin([y_min,y_max])]
plt.hist(data_y_desaturated, bins=100)  # plt.hist passes it's arguments to np.histogram
plt.title("Distribution of Desaturated Y")
plt.show()

'''
Distribution of desaturated Ys fit to Gaussian distribution well
'''
#%%
'''
explore ID
'''
data_id = explore_col('id')

'''
based on observation.
id is NOT the id of sample, it is id of subject (e.g. person, company, etc.) who produce sample.
the SUBJECT could produce sample at different time stamp
'''

#%%
'''
explore timestamp
'''
data_timestamp = explore_col('timestamp')

#%%
'''
explore ID and Timestamp
'''
data_id_timestamp = pd.concat([
    original_data['id'],
    original_data['timestamp']
    ],axis=1)

#%%
'''
check whether there is duplicate combination of ID and Timestamp
'''

id_plus_timestamp = data_id_timestamp['id'].apply(str)+"-"+data_id_timestamp['timestamp'].apply(str)

id_plus_timestamp_unq = id_plus_timestamp.unique()

print('Is [ID,Timestamp] unique?',id_plus_timestamp.size==id_plus_timestamp_unq.size,id_plus_timestamp.size,id_plus_timestamp_unq.size)

'''
Output
Is [ID,Timestamp] unique? True 1710756 1710756 

Observation
evey Subject only have one sample at every timestamp
'''

#%%
gps = data_id_timestamp.groupby('timestamp')

for key in gps.groups.keys():
    print('T'+key,end=',')
    gp_data = gps.groups[key]
    print(len(gp_data))
#%%
'''
explore features, generate statistics in excel sheet.
'''
import xlsxwriter
wb = xlsxwriter.Workbook('col_stats.xlsx')
ws = wb.add_worksheet()

header_row_pos = {'count':1,'min':2,'max':3,'mean':4,'std':5,'nan':6}
for row_header in header_row_pos.keys():
    row_pos = header_row_pos[row_header]
    ws.write(row_pos,0,row_header)

for col_id,col_name in enumerate(original_data.columns):
    col_pos = col_id+1
    print(col_name)
    ws.write(0,col_pos,col_name)
    col_data = original_data[col_name]
    ws.write_number(header_row_pos['count'],col_pos,len(col_data))
    ws.write_number(header_row_pos['min'],col_pos,np.min(col_data))
    ws.write_number(header_row_pos['max'],col_pos,np.max(col_data))
    ws.write_number(header_row_pos['mean'],col_pos,np.mean(col_data))
    ws.write_number(header_row_pos['std'],col_pos,np.std(col_data))
    ws.write_number(header_row_pos['nan'],col_pos,np.sum(np.isnan(col_data)))


wb.close()