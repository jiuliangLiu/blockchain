# -*- coding: utf-8 -*-
import base64
from Crypto.Cipher import AES
import re
import matplotlib.pyplot as plt
import itertools
import math
from function_test import aes_en, encodeToBinStr #导入加密等函数
base_read_path='G:\Python_Crawler\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path='G:\实验\秘密信息\message{}.txt'

def getParaCountList():
    paraCountList=[]
    for x in range(1,40):
        read_path=base_read_path.format(x)
    #    print(read_path)
    #    print(x)
        with open(read_path, 'r',encoding='utf-8') as fp:
            source_code=fp.read()
            #print(source_code) 
        matchFunctionPara=re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)',source_code)
#        print('第',x,'个合约匹配到的函数总数为：',len(matchFunctionPara))
        for var in matchFunctionPara:
            splitList=re.split(r'[,\()]',var)
            paraList=[]
            for index,x in enumerate(splitList):
    #            print(x)
                if index!=0 and index!=len(splitList)-1:
    #                print(x)
                    paraList.append(x.strip())
    #        print(paraList)
#            print("函数参数个数为：",len(paraList))
            paraCountList.append(len(paraList))
    return paraCountList
def getCoverOrderList():
    coverOrderList=[]
    for x in range(1,4001):
        read_path=base_read_path.format(x)
    #    print(read_path)
    #    print(x)
        with open(read_path, 'r',encoding='utf-8') as fp:
            source_code=fp.read()
            #print(source_code) 
        matchFunctionPara=re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)',source_code)
#        print('第',x,'个合约匹配到的函数总数为：',len(matchFunctionPara))
        for var in matchFunctionPara:
    #        print(var)
            splitList=re.split(r'[,\()]',var)
    #        for x in splitList:
    #            print(x)
    #        print(splitList)
    #        print(len(splitList))
#            paraAmount=len(splitList)-2 
    #        print(paraAmount)
            paraList=[]
            for index,x in enumerate(splitList):
    #            print(x)
                if index!=0 and index!=len(splitList)-1:
    #                print(x)
                    paraList.append(x.strip())
    #        print(paraList)
            paraTuple=tuple(paraList)
    #        print(paraTuple)
#            print("函数参数个数为：",len(paraList))
            if len(paraList)<10:
                paraPerm=list(itertools.permutations(paraList))
                paraPerm.sort()
    #            print(paraPerm)
#                print('函数参数的顺序特征值为：',paraPerm.index(paraTuple))
                coverOrderList.append(paraPerm.index(paraTuple))
    #        break
    #print(orderList)
    #print(len(orderList))
    return coverOrderList
def getStegoOrderList():
    stegoOrderList=[]
    for x in range(1,2):
        read_path=base_message_path.format(x)
        with open(read_path, 'r',encoding='utf-8') as fp:
            message=fp.read()
#            print(message)
        enMessage=encodeToBinStr(aes_en(message)).replace(' ','')
#            print(enMessage)
        paraCountList=getParaCountList()
        for var in paraCountList:
            varFactorial=math.factorial(var) #参数数目的阶乘
            varlog=math.log(varFactorial)/math.log(2) #阶乘的对数
#                print(varlog)
            bitCount=math.floor(varlog)
            stegoOrderList.append(int(enMessage[0:bitCount],base=2))
            enMessage=enMessage[bitCount:len(enMessage)]
#            print('加密后的消息长度为：',len(enMessage))
#            print('stego顺序编号为：',int(enMessage[0:bitCount],base=2))
    return stegoOrderList
def drawHist(orderList):
    for i in range(len(orderList))[::-1]:
        if orderList[i] > 200:
            del orderList[i]
    plt.figure()
    plt.hist(orderList, bins=200, color='r', alpha=0.9)
    plt.show()
if __name__=='__main__':
    coverOrderList=getCoverOrderList()
    drawHist(coverOrderList)
#    print('零的个数为：',coverOrderList.count(0))
#    print('一的个数为：',coverOrderList.count(1))
#    print('二的个数为：',coverOrderList.count(2))
    stegoOrderList=getStegoOrderList()
    drawHist(stegoOrderList)
#    print('零的个数为：',stegoOrderList.count(0))
#    print('一的个数为：',stegoOrderList.count(1))
#    print('二的个数为：',stegoOrderList.count(2))
#    print('248的个数为：',stegoOrderList.count(248))
#    paraCountList=getParaCountList()
#    print(paraCountList)