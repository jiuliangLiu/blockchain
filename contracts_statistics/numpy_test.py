# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:12:25 2019

@author: 刘九良
"""

import numpy as np

#ndarray各属性
#data=np.random.rand(2,3)
#print(data)
#print(type(data))
#print('维度个数：',data.ndim)
#print('各维度大小：',data.shape)
#print('数据类型：',data.dtype)

#list转ndarray
#l=range(9)
#data=np.array(l)
#print(data)
#print(data.shape)
#print(data.ndim)

#嵌套列表转换为ndarray
#l2=[range(10),range(10)]
#print(l2)
#data=np.array(l2)
#print(data)
#print(data.shape)

#np.zeros, np.ones 和 np.empty
#zeros_arr=np.zeros((3,4))
#print(zeros_arr)
#print(zeros_arr.dtype)
#ones_arr=np.ones((3,2))
#print(ones_arr)
#print(ones_arr.dtype)
#empty_arr=np.empty((3,3))
#print(empty_arr)
#empty_int_arr=np.empty((3,3),int)
#print(empty_int_arr)
#print(np.arange(10))

#zeros_float_arr=np.zeros((3,4),dtype=np.float64)
#print(zeros_float_arr)
#print(zeros_float_arr.dtype)

#矢量与矢量运算
#arr=np.array([[1,2,3],
#              [4,5,6]])
#print("元素相乘：")
#print(arr*arr)
#print("矩阵相加：")
#print(arr+arr)

#矢量与标量运算
#arr=np.array([[1,2,3],
#              [4,5,6]])
#print(1./arr)
#print(2.*arr)

#数组切片
#arr1=np.arange(10)
##print(arr1)
##print(arr1[:3])
#arr2=np.arange(12).reshape(3,4)
#print(arr2)
#print(arr2[1])
#print(arr2[0:2,2:])

#条件索引
#找出data_arr中2015年后的数据
#data_arr=np.random.rand(3,3)
#print(data_arr)
#year_arr=np.array([[2000,2001,2000],
#                  [2005,2002,2009],
#                  [2001,2003,2010]])
##filtered_arr=data_arr[year_arr>=2005]
##print(filtered_arr)
#filtered_arr=data_arr[(year_arr<=2005)&(year_arr%2==0)]
#print(filtered_arr)

#通用函数
#arr=np.random.randn(2,3)
#print(arr)
#print(np.ceil(arr))
#print(np.floor(arr))
#print(np.rint(arr))
#print(np.isnan(arr))

#np.where
#arr=np.random.randn(3,4)
#print(arr)
#npWh=np.where(arr>0,1,-1)
#print(npWh)

#常用的统计方法
arr=np.arange(10).reshape(5,2)
print(arr)
print(np.sum(arr))
print(np.sum(arr,axis=0))
print(np.sum(arr,axis=1))