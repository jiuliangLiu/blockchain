# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:40:58 2019

@author: 刘九良
"""

import re
import numpy as np
# import pandas as pd
from sklearn import svm
from function_test import aes_en, encodeToBinStr  # 导入加密等函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from svm_uint_test import word_count_in_str, geEnMessage
from svm_bBlock_test import splitList
import itertools
import math
import random

base_read_path = r'E:\刘九良\experimentData\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path = r'E:\刘九良\experimentData\message1.txt'  # 秘密信息存储路径


def get_mixRandom_feature():
    coverMixFeature = []  # 载体合约中的uint和uint256特征列表
    stegoMixFeature = []  # 含秘载体合约中的uint和uint256特征列表
    enMessage = geEnMessage()
    for x in range(1, 20000):
        if len(enMessage) < 100:
            print("enMessage已经嵌入完毕！！！")
            enMessage = geEnMessage()
        #        print('第',x,'个合约：')
        eachCoverFeature = []  # 存储每个cover合约的多维特征
        eachStegoFeature = []  # 存储每个stego合约的多维特征
        eachCoverFunFeature = []  # 每个cover中的函数特征列表
        eachStegoFunFeature = []  # 每个stego中的函数特征列表
        eachCoverFunParaFeature = []  # 每个cover中的函数参数特征列表
        eachStegoFunParaFeature = []  # 每个stego中的函数参数特征列表
        read_path = base_read_path.format(x)
        #    print(read_path)
        #    print(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        uintPattern = re.compile(r'\buint(256)?\b')
        matchUint = uintPattern.finditer(source_code)
        uintList = []
        for i, match in enumerate(matchUint):
            #            print("第",i,"个匹配到的值为：",match.group())
            uintList.append(match.group())
        coverUintLen = uintList.count('uint')
        #        print('coverUintLen为：',coverUintLen)
        coverUint256Len = uintList.count('uint256')
        #        print('coverUint256Len为：',coverUint256Len)
        totalLen = len(uintList)
        matchFunction = re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        if totalLen > 0 and len(matchFunction) > 1:  # 如果包含uint或uint256，则可以提出特征

            # 以下开始获取第一维特征
            #            eachCoverFeature.append((coverUint256Len-coverUintLen)/(float(coverUint256Len)+coverUintLen))
            #            #使用随机数种子选取嵌入点
            #            random.seed(9)
            #            uintDataList=list(range(int(totalLen)))
            #            uintInsertIndex=[] #选出嵌入索引
            #            for i in range(int(totalLen/2)):
            #                randIndex=int(random.uniform(0,len(uintDataList)))
            #                uintInsertIndex.append(uintDataList[randIndex])
            #                del(uintDataList[randIndex])
            ##            print("总长度为：",totalLen)
            #            print("insertIndex为",uintInsertIndex)
            ##            print("insertIndex的长度为",len(insertIndex))
            ##            print("dataList为",dataList)
            #            subEnMessage=enMessage[0:len(uintInsertIndex)]
            ##            print("subEnMessage为：",subEnMessage)
            #            for i,eachIndex in enumerate(uintInsertIndex):
            #                if subEnMessage[i]=='0':
            ##                    print("秘密消息为0")
            #                    uintList[eachIndex]='uint'
            #                else:
            ##                    print("秘密消息为1")
            #                    uintList[eachIndex]='uint256'
            ##            print("新的uintList为：",uintList)
            #            stegoUintLen=uintList.count('uint')
            #            stegoUint256Len=uintList.count('uint256')
            ##            print("新uint个数为：",uintList.count('uint'))
            ##            print("新uint256的个数为：",uintList.count('uint256'))
            #            print("uint处理前enMessage的原长度为：",len(enMessage))
            #            eachStegoFeature.append((stegoUint256Len-stegoUintLen)/(float(stegoUintLen)+stegoUint256Len)) #stego的uint特征
            #            enMessage=enMessage[len(uintInsertIndex):len(enMessage)] #获取剩余的消息
            #            print("uint处理后剩余的enMessage长度为：",len(enMessage))

            # 以下开始获取第二维特征
            functionNameList = []
            # 以下循环用于获取函数名列表
            for var in matchFunction:  # 获取函数名列表
                #        print(var)
                split_List = re.split(r'[,\()]', var)
                functionName = split_List[0].replace('function', '').strip()
                functionNameList.append(functionName)  # 获取函数名列表
            print("函数名列表的长度为：", len(functionNameList))
            funCount = len(functionNameList)
            groupCount = math.ceil(funCount / 8)
            print("函数列表可以分为", groupCount, "组")
            # 使用随机数种子决定是否嵌入数据
            funDataList = list(range(groupCount))
            funInsertIndex = []  # 选出嵌入索引
            for i in range(int(groupCount / 2)):
                randIndex = int(random.uniform(0, len(funDataList)))
                funInsertIndex.append(funDataList[randIndex])
                del (funDataList[randIndex])
            print("funInsertIndex为：", funInsertIndex)
            functionNameGroupList = splitList(functionNameList, 8)  # 将函数名分为8个一组
            # 以下循环对每组函数名进行处理
            for funGroupIndex, eachFunNameList in enumerate(functionNameGroupList):
                if len(eachFunNameList) > 1:  # 只处理长度大于1的函数名列表
                    funNameHashList = []  # 函数名哈希值列表
                    hashNameDict = {}  # 创建哈希值和name的对应关系字典
                    for eachFunName in eachFunNameList:
                        funNameHashList.append(hash(eachFunName))
                        hashNameDict[hash(eachFunName)] = eachFunName
                    functionNameTuple = tuple(eachFunNameList)
                    #                    print("原函数名的顺序为：",functionNameTuple)
                    #                    print("原函数名哈希值的顺序为：",functionNameHashTuple)
                    functionPerm = list(itertools.permutations(eachFunNameList))
                    functionPerm.sort()
                    eachCoverFunFeature.append(functionPerm.index(functionNameTuple))  # 载体的函数顺序特征
                    if funInsertIndex.count(funGroupIndex) == 0:
                        print("不嵌入秘密信息")
                        eachStegoFunFeature.append(functionPerm.index(functionNameTuple))
                    else:
                        print("嵌入秘密信息")
                        functionHashPerm = list(itertools.permutations(funNameHashList))  # 函数名的哈希值排列
                        functionHashPerm.sort()
                        print(type(functionHashPerm))
                        functionNumFactorial = math.factorial(len(eachFunNameList))  # 函数数数目的阶乘
                        functionNumLog = math.log(functionNumFactorial) / math.log(2)  # 阶乘的对数
                        #                print(varlog)
                        if functionNumLog - int(functionNumLog) == 0:
                            bitCount = int(functionNumLog)  # 可以存储的位数刚好为整数
                            permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                            print("可以存储1位", "位数：", bitCount)
                            print("排列的索引为：", permIndex)
                        else:  # 可以存储的位数不为整数，需要判断存储的位数
                            if int(enMessage[0:int(functionNumLog) + 1], base=2) > (2 ** int(functionNumLog) - 1):
                                bitCount = int(functionNumLog)
                                permIndex = int(enMessage[0:bitCount], base=2) + functionNumFactorial - 2 ** int(
                                    functionNumLog)  # 排列索引
                            #                            print("可以存储的位数不为整数,且不能多存一位,存储的位数为：",bitCount)
                            #                            print("排列索引为：",permIndex)
                            else:
                                bitCount = int(functionNumLog) + 1
                                permIndex = int(enMessage[0:bitCount], base=2)
                        #                            print("可以存储的位数不为整数,且能多存一位,存储的位数为：",bitCount)
                        #                            print("排列索引为：",permIndex)
                        #                    print("索引值为:",permIndex)
                        #                    print("函数名哈希值排列列表的长度为：",len(functionHashPerm))
                        funNameHashAfterPerm = list(functionHashPerm[permIndex])  # 排列之后的函数名哈希值列表
                        #                    print("排列后的函数的哈希值顺序为：",funNameHashAfterPerm)
                        funNameAfterPerm = []  # 排列之后的函数名列表
                        for funNameHash in funNameHashAfterPerm:
                            funNameAfterPerm.append(hashNameDict[funNameHash])
                            #                    print("对应的原函数名为:",funNameAfterPerm)
                        #                    print("排列后的函数的顺序特征为:",functionPerm.index(tuple(funNameAfterPerm)))
                        eachStegoFunFeature.append(functionPerm.index(tuple(funNameAfterPerm)))
                        #                    print("之前求的顺序特征为：",permIndex)
                        #            stegoFunctionFeature.append(permIndex) #stego的函数顺序特征
                        print("function处理前enMessage的长度为：", len(enMessage))
                        enMessage = enMessage[bitCount:len(enMessage)]
                        print("function处理后enMessage的长度为：", len(enMessage))
            strCoverFeature = ''
            strStegoFeature = ''
            for var in eachCoverFunFeature:
                strCoverFeature += bin(var)[2:len(bin(var))]
            #            print("strCoverFeature为：",strCoverFeature)
            #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
            for var in eachStegoFunFeature:
                strStegoFeature += bin(var)[2:len(bin(var))]
            #            print("steStegoFeature为：",strStegoFeature)
            #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
            # 每个合约的第二维特征
            eachCoverFeature.append(int(strCoverFeature, base=2))
            eachStegoFeature.append(int(strStegoFeature, base=2))

            # 开始获取第三维特征，以下循环获取每个合约的函数参数特征
            funParaDataList = list(range(len(matchFunction)))
            funParaInsertIndex = []  # 选出嵌入索引
            for i in range(int(len(matchFunction) / 2)):
                randIndex = int(random.uniform(0, len(funParaDataList)))
                funParaInsertIndex.append(funParaDataList[randIndex])
                del (funParaDataList[randIndex])
            print("funParaInsertIndex为：", funParaInsertIndex)
            for funIndex, eachFun in enumerate(matchFunction):
                #        print(var)
                if funIndex > 8:  # 只获取前8个函数的参数特征
                    break
                split_List = re.split(r'[,\()]', eachFun)
                paraList = []  # 获取每个函数的参数列表
                paraHashList = []  # 获取每个函数参数的哈希值列表
                hashParaDict = {}  # 函数参数和哈希值的对应字典
                for funParaIndex, funPara in enumerate(split_List):
                    #            print(funPara)
                    if funParaIndex != 0 and funParaIndex != len(split_List) - 1:
                        #                print(funPara)
                        paraList.append(funPara.strip())
                        paraHashList.append(hash(funPara.strip()))
                        hashParaDict[hash(funPara.strip())] = funPara.strip()
                if 1 < len(paraList) < 10:  # 如果参数大于等于2个
                    paraTuple = tuple(paraList)
                    #                    print(paraTuple)
                    paraPerm = list(itertools.permutations(paraList))
                    paraPerm.sort()  # 排列之后进行排序
                    paraHashPerm = list(itertools.permutations(paraHashList))
                    paraHashPerm.sort()
                    #            print(paraPerm)
                    #                    print('函数参数的顺序特征值为：',paraPerm.index(paraTuple))
                    eachCoverFunParaFeature.append(paraPerm.index(paraTuple))
                    print("索引值为：", funIndex)
                    if funParaInsertIndex.count(funIndex) == 0:
                        print("不嵌入秘密信息")
                        eachStegoFunParaFeature.append(paraPerm.index(paraTuple))
                    else:
                        print("嵌入秘密信息")
                        varFactorial = math.factorial(len(paraList))  # 函数参数数目的阶乘
                        varlog = math.log(varFactorial) / math.log(2)  # 阶乘的对数
                        #                print(varlog)
                        if varlog - int(varlog) == 0:
                            bitCount = int(varlog)  # 可以存储的位数刚好为整数
                            permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                        #                        print("可以存储1位","位数：",bitCount)
                        #                        print("排列的索引为：",permIndex)
                        else:  # 可以存储的位数不为整数，需要判断存储的位数
                            if int(enMessage[0:int(varlog) + 1], base=2) > (2 ** int(varlog) - 1):
                                bitCount = int(varlog)
                                permIndex = int(enMessage[0:bitCount], base=2) + varFactorial - 2 ** int(varlog)  # 排列索引
                            #                            print("可以存储的位数不为整数,且不能多存一位,存储的位数为：",bitCount)
                            #                            print("排列索引为：",permIndex)
                            else:
                                bitCount = int(varlog) + 1
                                permIndex = int(enMessage[0:bitCount], base=2)
                        paraHashAfterPerm = list(paraHashPerm[permIndex])
                        paraAfterPerm = []  # 排列之后的函数参数名列表
                        for paraHash in paraHashAfterPerm:
                            paraAfterPerm.append(hashParaDict[paraHash])
                            #                    print("排列后的参数哈希值列表为：",paraHashAfterPerm)
                        #                    print("排列后的参数列表为：",paraAfterPerm)
                        #                    print("排列后的函数的顺序特征为:",paraPerm.index(tuple(paraAfterPerm)))
                        eachStegoFunParaFeature.append(paraPerm.index(tuple(paraAfterPerm)))
                        print("funPara处理前的enMessage长度为：", len(enMessage))
                        enMessage = enMessage[bitCount:len(enMessage)]
                        print("funPara处理后的enMessage长度为：", len(enMessage))
            if len(eachCoverFunParaFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverFunParaFeature:
                    strCoverFeature += bin(var)[2:len(bin(var))]
                #                print("strCoverFeature为：",strCoverFeature)
                #                print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoFunParaFeature:
                    strStegoFeature += bin(var)[2:len(bin(var))]
                #                print("strStegoFeature为：",strStegoFeature)
                #                print("eachStegoFeature为：",int(strStegoFeature,base=2))
                # 每个合约的第三维特征
                eachCoverFeature.append(int(strCoverFeature, base=2))
                eachStegoFeature.append(int(strStegoFeature, base=2))

            # 开始获取第四维特征
            equalStatepattern = re.compile(r'\b([\w.\[\]]+?)\s*(>=|<=|>|<|==|!=|\+=|-=)\s*([\w.\[\]]+?)\b')
            matchEqualState = equalStatepattern.findall(source_code)
            zeroLen = 0
            oneLen = 0
            totalLen = len(matchEqualState)
            if totalLen > 0:
                for var in matchEqualState:
                    #            print(len(var))
                    #                print("var[0]为：",var[0])
                    #                print("var[2]为：",var[2])
                    if hash(var[0]) < hash(var[2]):
                        zeroLen = zeroLen + 1
                    else:
                        oneLen = oneLen + 1
                #            print("zeroLen",zeroLen)
                #            print("oneLen",oneLen)
                eachCoverFeature.append((zeroLen - oneLen) / (float(zeroLen) + oneLen))
                #        print("coverEqualSubFeature为：",coverEqualSubFeature)
                # 使用随机数种子选取嵌入点
                random.seed(9)
                equalSubDataList = list(range(int(totalLen)))
                equalSubInsertIndex = []  # 选出嵌入索引
                for i in range(int(totalLen / 2)):
                    randIndex = int(random.uniform(0, len(equalSubDataList)))
                    equalSubInsertIndex.append(equalSubDataList[randIndex])
                    del (equalSubDataList[randIndex])
                print("insertIndex为", equalSubInsertIndex)
                print("insertIndex的长度为：", len(equalSubInsertIndex))
                subEnMessage = enMessage[0:int(len(equalSubInsertIndex))]
                # 根据随机数种子计算stego特征
                stegoZeroLen = 0
                stegoOneLen = 0
                subMessageIndex = 0
                for stateIndex, var in enumerate(matchEqualState):
                    #                print("第",stateIndex,"条语句")
                    if equalSubInsertIndex.count(stateIndex) == 0:  # 不嵌入信息
                        #                    print("不嵌入信息")
                        if hash(var[0]) < hash(var[2]):
                            stegoZeroLen = stegoZeroLen + 1
                        else:
                            stegoOneLen = stegoOneLen + 1
                    else:  # 嵌入秘密信息
                        #                    print("嵌入信息")
                        if subEnMessage[subMessageIndex] == '0':
                            #                        print("嵌入0")
                            stegoZeroLen = stegoZeroLen + 1
                            subMessageIndex = subMessageIndex + 1
                        else:
                            #                        print("嵌入1")
                            stegoOneLen = stegoOneLen + 1
                            subMessageIndex = subMessageIndex + 1
                #            print("stegoZeroLen为：",stegoZeroLen)
                #            print("stegoOneLen为：",stegoOneLen)
                print("enMessage的原长度为：", len(enMessage))
                eachStegoFeature.append(
                    (stegoZeroLen - stegoOneLen) / (float(stegoZeroLen) + stegoOneLen))  # stego的uint特征
                enMessage = enMessage[int(len(equalSubInsertIndex)):len(enMessage)]  # 获取剩余的消息
                print("剩余的enMessage长度为", len(enMessage))
            if len(eachCoverFeature) == 3:  # 只需要四维特征
                coverMixFeature.append(eachCoverFeature)
                stegoMixFeature.append(eachStegoFeature)
    return coverMixFeature, stegoMixFeature


def getMixFeature():  # 分别获取cover和stego的Mix特征
    coverMixFeature = []  # 载体合约中的uint和uint256特征列表
    stegoMixFeature = []  # 含秘载体合约中的uint和uint256特征列表
    enMessage = geEnMessage()
    for x in range(1, 20000):
        if len(enMessage) < 100:
            print("enMessage已经嵌入完毕！！！")
            enMessage = geEnMessage()
        eachCoverFeature = []  # 存储每个cover合约的多维特征
        eachStegoFeature = []  # 存储每个stego合约的多维特征
        eachCoverFunFeature = []  # 每个cover中的函数特征列表
        eachStegoFunFeature = []  # 每个stego中的函数特征列表
        eachCoverFunParaFeature = []  # 每个cover中的函数参数特征列表
        eachStegoFunParaFeature = []  # 每个stego中的函数参数特征列表
        read_path = base_read_path.format(x)
        #    print(read_path)
        #    print(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        uintPattern = re.compile(r'\buint(256)?\b')
        matchUint = uintPattern.finditer(source_code)
        uintList = []
        for i, match in enumerate(matchUint):
            #            print("第",i,"个匹配到的值为：",match.group())
            uintList.append(match.group())
        matchFunction = re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        #        print('第',x,'个合约：')
        coverUintLen = uintList.count('uint')
        #        print('coverUintLen为：',coverUintLen)
        coverUint256Len = uintList.count('uint256')
        #        print('coverUint256Len为：',coverUint256Len)
        totalLen = len(uintList)
        if totalLen > 0 and len(matchFunction) > 1:  # 如果包含uint或uint256，则可以提出特征

            # cover的第一维特征
            #            eachCoverFeature.append((coverUint256Len-coverUintLen)/(float(coverUint256Len)+coverUintLen))
            #            subEnMessage=enMessage[0:int(totalLen)]
            ##            print("要去除的长度为：",int(totalLen),len(subEnMessage))
            #            print("uint处理前enMessage的原长度为：",len(enMessage))
            #            stegoUintLen=word_count_in_str(subEnMessage,'0')
            ##            print("stegoUintLen为：",stegoUintLen)
            #            stegoUint256Len=word_count_in_str(subEnMessage,'1')
            ##            print("stegoUint256Len为：",stegoUint256Len)
            #            #stego的第一维特征
            #            eachStegoFeature.append((stegoUint256Len-stegoUintLen)/(float(coverUint256Len)+coverUintLen))
            #            enMessage=enMessage[int(totalLen):len(enMessage)] #获取剩余的消息
            #            print("uint处理后剩余的enMessage长度为",len(enMessage))

            # 以下开始获取第二维特征
            functionNameList = []
            # 以下循环用于获取函数名列表
            for var in matchFunction:  # 获取函数名列表
                #        print(var)
                split_List = re.split(r'[,\()]', var)
                functionName = split_List[0].replace('function', '').strip()
                functionNameList.append(functionName)  # 获取函数名列表
            functionNameGroupList = splitList(functionNameList, 8)  # 将函数名分为8个一组
            # 以下循环对每组函数名进行处理
            for eachFunNameList in functionNameGroupList:
                if len(eachFunNameList) > 1:  # 只处理长度大于1的函数名列表
                    funNameHashList = []  # 函数名哈希值列表
                    hashNameDict = {}  # 创建哈希值和name的对应关系字典
                    for eachFunName in eachFunNameList:
                        funNameHashList.append(hash(eachFunName))
                        hashNameDict[hash(eachFunName)] = eachFunName
                    functionNameTuple = tuple(eachFunNameList)
                    #                    print("原函数名的顺序为：",functionNameTuple)
                    #                    print("原函数名哈希值的顺序为：",functionNameHashTuple)
                    functionPerm = list(itertools.permutations(eachFunNameList))
                    functionPerm.sort()
                    eachCoverFunFeature.append(functionPerm.index(functionNameTuple))  # 载体的函数顺序特征
                    functionHashPerm = list(itertools.permutations(funNameHashList))  # 函数名的哈希值排列
                    functionHashPerm.sort()
                    #            print(type(functionHashPerm))
                    functionNumFactorial = math.factorial(len(eachFunNameList))  # 函数数数目的阶乘
                    functionNumLog = math.log(functionNumFactorial) / math.log(2)  # 阶乘的对数
                    #                print(varlog)
                    if functionNumLog - int(functionNumLog) == 0:
                        bitCount = int(functionNumLog)  # 可以存储的位数刚好为整数
                        permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                    #                        print("可以存储1位","位数：",bitCount)
                    #                        print("排列的索引为：",permIndex)
                    else:  # 可以存储的位数不为整数，需要判断存储的位数
                        if int(enMessage[0:int(functionNumLog) + 1], base=2) > (2 ** int(functionNumLog) - 1):
                            bitCount = int(functionNumLog)
                            permIndex = int(enMessage[0:bitCount], base=2) + functionNumFactorial - 2 ** int(
                                functionNumLog)  # 排列索引
                        #                            print("可以存储的位数不为整数,且不能多存一位,存储的位数为：",bitCount)
                        #                            print("排列索引为：",permIndex)
                        else:
                            bitCount = int(functionNumLog) + 1
                            permIndex = int(enMessage[0:bitCount], base=2)
                    #                            print("可以存储的位数不为整数,且能多存一位,存储的位数为：",bitCount)
                    #                            print("排列索引为：",permIndex)
                    #                    print("索引值为:",permIndex)
                    #                    print("函数名哈希值排列列表的长度为：",len(functionHashPerm))
                    funNameHashAfterPerm = list(functionHashPerm[permIndex])  # 排列之后的函数名哈希值列表
                    #                    print("排列后的函数的哈希值顺序为：",funNameHashAfterPerm)
                    funNameAfterPerm = []  # 排列之后的函数名列表
                    for funNameHash in funNameHashAfterPerm:
                        funNameAfterPerm.append(hashNameDict[funNameHash])
                    #                    print("对应的原函数名为:",funNameAfterPerm)
                    #                    print("排列后的函数的顺序特征为:",functionPerm.index(tuple(funNameAfterPerm)))
                    eachStegoFunFeature.append(functionPerm.index(tuple(funNameAfterPerm)))
                    #                    print("之前求的顺序特征为：",permIndex)
                    #            stegoFunctionFeature.append(permIndex) #stego的函数顺序特征
                    print("function处理前enMessage的长度为：", len(enMessage))
                    enMessage = enMessage[bitCount:len(enMessage)]
                    print("function处理后enMessage的长度为：", len(enMessage))
            strCoverFeature = ''
            strStegoFeature = ''
            for var in eachCoverFunFeature:
                strCoverFeature += bin(var)[2:len(bin(var))]
            #            print("strCoverFeature为：",strCoverFeature)
            #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
            for var in eachStegoFunFeature:
                strStegoFeature += bin(var)[2:len(bin(var))]
            #            print("steStegoFeature为：",strStegoFeature)
            #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
            # 每个合约的第二维特征
            eachCoverFeature.append(int(strCoverFeature, base=2))
            eachStegoFeature.append(int(strStegoFeature, base=2))
            # 以下循环获取每个合约的函数参数特征
            for index, eachFun in enumerate(matchFunction):
                #        print(var)
                if index > 8:  # 只获取前8个函数的参数特征
                    break
                split_List = re.split(r'[,\()]', eachFun)
                functionName = split_List[0].replace('function', '').strip()
                functionNameList.append(functionName)  # 获取函数名列表
                paraList = []  # 获取每个函数的参数列表
                paraHashList = []  # 获取每个函数参数的哈希值列表
                hashParaDict = {}  # 函数参数和哈希值的对应字典
                for index, funPara in enumerate(split_List):
                    #            print(funPara)
                    if index != 0 and index != len(split_List) - 1:
                        #                print(funPara)
                        paraList.append(funPara.strip())
                        paraHashList.append(hash(funPara.strip()))
                        hashParaDict[hash(funPara.strip())] = funPara.strip()
                if 1 < len(paraList) < 10:  # 如果参数大于等于2个
                    paraTuple = tuple(paraList)
                    #                    print(paraTuple)
                    paraPerm = list(itertools.permutations(paraList))
                    paraPerm.sort()  # 排列之后进行排序
                    paraHashPerm = list(itertools.permutations(paraHashList))
                    paraHashPerm.sort()
                    #            print(paraPerm)
                    #                    print('函数参数的顺序特征值为：',paraPerm.index(paraTuple))
                    eachCoverFunParaFeature.append(paraPerm.index(paraTuple))
                    varFactorial = math.factorial(len(paraList))  # 函数参数数目的阶乘
                    varlog = math.log(varFactorial) / math.log(2)  # 阶乘的对数
                    #                print(varlog)
                    if varlog - int(varlog) == 0:
                        bitCount = int(varlog)  # 可以存储的位数刚好为整数
                        permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                    #                        print("可以存储1位","位数：",bitCount)
                    #                        print("排列的索引为：",permIndex)
                    else:  # 可以存储的位数不为整数，需要判断存储的位数
                        if int(enMessage[0:int(varlog) + 1], base=2) > (2 ** int(varlog) - 1):
                            bitCount = int(varlog)
                            permIndex = int(enMessage[0:bitCount], base=2) + varFactorial - 2 ** int(varlog)  # 排列索引
                        #                            print("可以存储的位数不为整数,且不能多存一位,存储的位数为：",bitCount)
                        #                            print("排列索引为：",permIndex)
                        else:
                            bitCount = int(varlog) + 1
                            permIndex = int(enMessage[0:bitCount], base=2)
                    paraHashAfterPerm = list(paraHashPerm[permIndex])
                    paraAfterPerm = []  # 排列之后的函数参数名列表
                    for paraHash in paraHashAfterPerm:
                        paraAfterPerm.append(hashParaDict[paraHash])
                    #                    print("排列后的参数哈希值列表为：",paraHashAfterPerm)
                    #                    print("排列后的参数列表为：",paraAfterPerm)
                    #                    print("排列后的函数的顺序特征为:",paraPerm.index(tuple(paraAfterPerm)))
                    eachStegoFunParaFeature.append(paraPerm.index(tuple(paraAfterPerm)))
                    print("funPara处理前的enMessage长度为：", len(enMessage))
                    enMessage = enMessage[bitCount:len(enMessage)]
                    print("funPara处理后的enMessage长度为：", len(enMessage))
            if len(eachCoverFunParaFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverFunParaFeature:
                    strCoverFeature += bin(var)[2:len(bin(var))]
                #                print("strCoverFeature为：",strCoverFeature)
                #                print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoFunParaFeature:
                    strStegoFeature += bin(var)[2:len(bin(var))]
                #                print("strStegoFeature为：",strStegoFeature)
                #                print("eachStegoFeature为：",int(strStegoFeature,base=2))
                # 每个合约的第三维特征
                eachCoverFeature.append(int(strCoverFeature, base=2))
                eachStegoFeature.append(int(strStegoFeature, base=2))

            # 开始获取第四维特征
            equalStatepattern = re.compile(r'\b([\w.\[\]]+?)\s*(>=|<=|>|<|==|!=|\+=|-=)\s*([\w.\[\]]+?)\b')
            matchEqualState = equalStatepattern.findall(source_code)
            zeroLen = 0
            oneLen = 0
            for var in matchEqualState:
                #                print(len(var))
                #                print("var[0]为：",var[0])
                #                print("var[2]为：",var[2])
                if hash(var[0]) < hash(var[2]):
                    zeroLen = zeroLen + 1
                else:
                    oneLen = oneLen + 1
            totalLen = len(matchEqualState)
            print("totalLen为：", totalLen)
            if totalLen > 0:
                eachCoverFeature.append((zeroLen - oneLen) / (float(zeroLen) + oneLen))
                #        print("coverEqualSubFeature为：",coverEqualSubFeature)
                subEnMessage = enMessage[0:int(totalLen)]
                print("equalSub处理前enMessage的长度为：", len(enMessage))
                stegoZeroLen = word_count_in_str(subEnMessage, '0')
                stegoOneLen = word_count_in_str(subEnMessage, '1')
                eachStegoFeature.append(
                    (stegoZeroLen - stegoOneLen) / (float(stegoZeroLen) + stegoOneLen))  # stego的uint特征
                enMessage = enMessage[int(totalLen):len(enMessage)]  # 获取剩余的消息
                print("equalSub处理后enMessage的长度为", len(enMessage))
            if len(eachCoverFeature) == 3:  # 只需要四维特征
                coverMixFeature.append(eachCoverFeature)
                stegoMixFeature.append(eachStegoFeature)
    return coverMixFeature, stegoMixFeature


def train():
    #    coverUintFeature,stegoUintFeature=get_mixRandom_feature()
    coverUintFeature, stegoUintFeature = getMixFeature()
    X = coverUintFeature + stegoUintFeature  # 训练特征
    X = np.array(X).reshape(-1, 3)
    #    print("总特征集为：",X)
    #    print("总特征集的长度为：",len(X))
    y = [0] * len(coverUintFeature) + [1] * len(stegoUintFeature)  # 训练标签
    #    print("对应的总标签为：",y)
    #    print("对应的总标签的长度为：",len(y))
    train_X, test_X, train_y, test_true_y = train_test_split(X, y, test_size=0.3, random_state=0)
    #    print("训练集的特征为：",train_X)
    #    print("测试集的特征为：",test_X)
    #    print("训练集的标签为：",train_y)
    #    print("测试集的标签真实值为：",test_true_y)
    #    print(type(test_true_y))
    #    train_X=np.array(train_X).reshape(-1,1)
    #    test_X=np.array(test_X).reshape(-1,1)

    # 线性核预测
    #    svm_model_linear=svm.SVC(kernel='linear',C=100.) #线性核
    #    svm_model_linear.fit(train_X,train_y)
    #    test_predict_linear_y=svm_model_linear.predict(test_X) #预测的真实值
    #    print("SVM线性核预测的准确度为：",accuracy_score(test_true_y,test_predict_linear_y))

    # rbf核预测，网格搜索最优参数
    #    model=svm.SVC(kernel='rbf')
    #    c_can=np.linspace(109,110,10)
    #    gamma_can = np.linspace(0.22, 0.24, 10)
    #    svc_rbf = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
    #    svc_rbf.fit(train_X,train_y)
    #    print ('最优参数：\n', svc_rbf.best_params_)

    # rbf核预测
    svm_model_rbf = svm.SVC(kernel='rbf', gamma=0.235, C=109.)  # 高斯核
    svm_model_rbf.fit(train_X, train_y)
    test_predict_rbf_y = svm_model_rbf.predict(test_X)  # 预测的真实值
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))


# 画图
#    n_support_vector=svm_model_rbf.n_support_ #支持向量的个数
#    print("支持向量的个数为：",n_support_vector)
#    Support_vector_index = svm_model_rbf.support_ #支持向量索引
#    print("支持向量的索引为：",Support_vector_index)
#    plot_point(train_X,train_y,Support_vector_index)

def plot_point(dataArr, labelArr, Support_vector_index):
    print("dataArr为：", dataArr)
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)
    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == '__main__':
    train()
#    coverMixFeature,stegoMixFeature=getMixFeature()
#    coverMixFeature,stegoMixFeature=get_mixRandom_feature()
#    coverMixFeature=np.array(coverMixFeature).reshape(-1,4)
#    stegoMixFeature=np.array(stegoMixFeature).reshape(-1,4)
#    print("coverMixFeature为：",coverMixFeature,"长度为：",len(coverMixFeature))
#    print("stegoMixFeature为：",stegoMixFeature,"长度为：",len(stegoMixFeature))
