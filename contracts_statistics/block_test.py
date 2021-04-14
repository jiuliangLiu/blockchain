# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:46:16 2019

@author: 刘九良
"""

import re
import random
from svm_bBlock_test import get_PC_Inst_GhiDict,splitGhiList,splitInstDict2

base_read_path='G:\coverContract\\binRuntime\\binRuntime{}.bin-runtime'

def getCoverAverageSize():
    totalSize=0
    for x in range(1,13212):
        read_path=base_read_path.format(x)
        with open(read_path, 'r',encoding='utf-8') as fp: #获取秘密信息
            binRunCode=fp.read()
        print("第",x,"个runtime字节码的大小为：",len(binRunCode)/2,"bytes")
        totalSize+=len(binRunCode)/2
    print("总大小为：",totalSize,"字节")
    averageSize=totalSize/13212
    print("平均大小为：",averageSize,"字节")
    
def getStegoAverageSize():
    totalAddSize=0
    for x in range(1,13213):
        read_path=base_read_path.format(x)
        with open(read_path, 'r',encoding='utf-8') as fp: #获取秘密信息
            binRunCode=fp.read()
        if binRunCode.strip(): #如果匹配到的binRuntime文件不为空
            #去掉尾部swarm哈希代码
            binRunCode=binRunCode[0:binRunCode.find('a165627a7a72305820')]
            #获取基本块开始索引
            index575b=binRunCode.find('575b')
#            print("575b的开始位置为：",index575b)
            #获取所有基本块代码
            allBlockCode=binRunCode[index575b+2:len(binRunCode)]
            allBlockGhiDic=get_PC_Inst_GhiDict(allBlockCode)
            allBlockGhiCode=''
            for eachGhiInst in allBlockGhiDic.values():
                allBlockGhiCode=allBlockGhiCode+eachGhiInst
            blockGhiList=re.findall(r'[I\dabcdef][^GHIJKL]*[GHJKL\dabcdef]',allBlockGhiCode)
            
            #获取基本块列表
            blockList=[]
            for blockGhiIndex,eachGhiBlock in enumerate(blockGhiList):
#                print("第",blockGhiIndex+1,"个Ghi基本块为：",eachGhiBlock)
                eachBlock=eachGhiBlock
                for eachGhi in splitGhiList:
                    if eachGhiBlock.count(eachGhi)!=0:
                        eachBlock=eachBlock.replace(eachGhi,splitInstDict2[eachGhi])
                blockList.append(eachBlock)  
            print("第",x,"个合约匹配到的基本块数目为：",len(blockList))
            print("第",x,"个合约增加的大小为：",int(len(blockList)/2))
            totalAddSize+=int(len(blockList)/2)
    print("总增加的大小为：",totalAddSize)
    print("平均增加的大小为：",int(totalAddSize/13212))
    print("stego合约的大小为：",2767+int(totalAddSize/13212))
    
def getTime():
    x=random.uniform(13, 14)
    print(x)

if __name__ == '__main__':
    getTime()