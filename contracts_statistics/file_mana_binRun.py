# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:21:49 2019

@author: 刘九良
"""
from glob import glob
import os

base_write_path='G:\coverContract\\binRuntime'
binRunNames=glob(r"H:\solidity\evmBin\*.bin-runtime") #获取.bin-runtime文件名列表
for i,fileName in enumerate(binRunNames):
#    if i<1:
#        continue
    newIndex=i+1
#    print("第",i,"个文件的文件绝对路径为：",fileName)
    with open(fileName,'r',encoding='utf-8') as fp:
        binRun_code=fp.read()
#    print("第",i,"个文件的文件内容为：",binRun_code)
    name=os.path.basename(fileName)
#    print("第",i,"个文件的文件名为：",name)
    write_path=base_write_path+"\\"+"binRuntime"+str(newIndex)+".bin-runtime" #写绝对路径
    print("第",i,"个文件的写路径为：",write_path)
    with open(write_path,'w',encoding='utf-8') as fp:
            fp.write(binRun_code)
#    if i>10:
#        break
