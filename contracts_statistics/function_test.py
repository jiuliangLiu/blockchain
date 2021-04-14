# -*- coding: utf-8 -*-
#测试载体中函数的顺序特征和stego中函数的顺序特征的差异
import base64
from Crypto.Cipher import AES
import re
import matplotlib.pyplot as plt
import itertools
import math
base_read_path='G:\coverContract\coverContract(no annotation)\coverContract{}.sol'
base_message_path='G:\实验\秘密信息\message{}.txt'

def encodeToBinStr(s): #将字符串解码成二进制字符串
    return ' '.join([bin(ord(c)).replace('0b', '') for c in s])

def aes_en(text): #AES加密函数
    key = '8bws41p1x6qml4.1'  # 加密秘钥要设置16位
    length =16
    count = len(text.encode('utf-8'))
    # text不是16的倍数那就补足为16的倍数
    if (count % length != 0):
        add = length - (count % length)
    else:
        add = 0
    entext = text + ('\0' * add)
 
    # 初始化加密器
    aes = AES.new(str.encode(key), AES.MODE_ECB)
    enaes_text = str(base64.b64encode(aes.encrypt(str.encode(entext))),encoding='utf-8')
#    enaes_text = aes.encrypt(str.encode(entext))
    return enaes_text
def getFunctionCountList():
    functionCountList=[]
    for x in range(1,10001):
        read_path=base_read_path.format(x)
    #    print(read_path)
    #    print(x)
        with open(read_path, 'r',encoding='utf-8') as fp:
            source_code=fp.read()
            #print(source_code) 
        matchFunction=re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)',source_code)
#        print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
        if 1<len(matchFunction)<10:
            functionCountList.append(len(matchFunction)) 
#    print(functionCountList)
    return functionCountList
def getCoverOrderList(): #获取函数顺序特征列表
    functionOrderList=[]
    for x in range(1,2000):
        functionNameList=[]
        read_path=base_read_path.format(x)
        with open(read_path, 'r',encoding='utf-8') as fp:
            source_code=fp.read()
            #print(source_code) 
        matchFunction=re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)',source_code)
#        print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
        if 1<len(matchFunction)<10:#匹配到的函数个数大于1小于10
            print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
            for var in matchFunction:
        #        print(var)
                splitList=re.split(r'[,\()]',var)
                functionNameList.append(splitList[0].replace('function','').strip())#获取函数名列表
    #        print('第',x,'个合约的函数名列表为：',functionNameList)
#        for i in range(len(functionNameList))[::-1]:
#            if len(functionNameList)>9:
#                del functionNameList[i]   
#        print('新长度为：',len(functionNameList))
            functionNameTuple=tuple(functionNameList)
            print(functionNameTuple)
            functionPerm=list(itertools.permutations(functionNameList))
            functionPerm.sort()
            functionOrderList.append(functionPerm.index(functionNameTuple))
    return functionOrderList
def getStegoOrderList():
    stegoOrderList=[]
    for x in range(1,2):
        read_path=base_message_path.format(x)
        with open(read_path, 'r',encoding='utf-8') as fp:
            message=fp.read()
#            print(message)
        enMessage=encodeToBinStr(aes_en(message)).replace(' ','')
        functionCountList=getFunctionCountList()
        for var in functionCountList:
            varFactorial=math.factorial(var) #函数数数目的阶乘
            varlog=math.log(varFactorial)/math.log(2) #阶乘的对数
#                print(varlog)
            bitCount=math.floor(varlog)
            stegoOrderList.append(int(enMessage[0:bitCount],base=2))
            enMessage=enMessage[bitCount:len(enMessage)]
#            print('加密后的消息长度为：',len(enMessage))
#            print('stego顺序编号为：',int(enMessage[0:bitCount],base=2))
#    print(stegoOrderList)
    return stegoOrderList
def drawHist(orderList):
    plt.figure()
    plt.hist(orderList, bins=200, color='r', alpha=0.9)
    plt.show()
if __name__=='__main__':
    functionOrderList=getCoverOrderList()
    drawHist(functionOrderList)
    stegoOrderList=getStegoOrderList()
    drawHist(stegoOrderList)