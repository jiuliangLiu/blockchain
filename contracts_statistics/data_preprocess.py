# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:45:42 2019

@author: 刘九良
"""

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np

#数据标准化
#X = np.array([[1., -1., 2.],
#              [2., 0., 0.],
#              [0., 1., -1.]])
#ss = StandardScaler()
#print(X)
#scaler = ss.fit(X) # <class 'sklearn.preprocessing.data.StandardScaler'>
## print(ss is scaler) # True
##print(scaler)
##print(scaler.mean_)
#
#transform = scaler.transform(X)
#print(transform)

#数据归一化
##归一到[0,1]
#x = np.array([[3., -1., 2., 613.],
#              [2., 0., 0., 232],
#              [0., 1., -1., 113],
#              [1., 2., -3., 489]])
#min_max_scaler = preprocessing.MinMaxScaler()
#x_minmax = min_max_scaler.fit_transform(x)
#print(x_minmax)

##归一到[-1,1]
x = np.array([[3., -1., 2., 613.],
              [2., 0., 0., 232],
              [0., 1., -1., 113],
              [1., 2., -3., 489]])
max_abs_scaler = preprocessing.MaxAbsScaler()
x_train_maxsbs = max_abs_scaler.fit_transform(x)
print(x_train_maxsbs)

