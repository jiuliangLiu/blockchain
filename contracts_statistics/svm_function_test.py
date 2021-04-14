# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:49:44 2019
用SVM检测函数排序隐藏信息
@author: 刘九良
"""
import re
import math
import itertools
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from function_test import aes_en, encodeToBinStr
from sklearn.model_selection import train_test_split
from svm_bBlock_test import splitList
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import random
from svm_uint_test import drawHist

base_read_path = r'F:\experimentData\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path = r'F:\experimentData\message1.txt'  # 秘密信息存储路径


def get_funRandom_Feature():
    coveFunctionFeature = []  # 载体合约中的函数特征列表
    stegoFunctionFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 20000):
        eachCoverConFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoConFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchFunction = re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        if len(matchFunction) > 1:  # 匹配到的函数数目大于1
            #            print("匹配到的函数数目为：",len(matchFunction))
            functionNameList = []
            for var in matchFunction:  # 获取函数名列表
                #        print(var)
                split_List = re.split(r'[,\()]', var)
                functionName = split_List[0].replace('function', '').strip()
                functionNameList.append(functionName)  # 获取函数名列表
            #            print("函数名列表的长度为：",len(functionNameList))
            funCount = len(functionNameList)
            groupCount = math.ceil(funCount / 8)
            #            print("函数列表可以分为",groupCount,"组")

            # 使用随机数种子决定是否嵌入数据
            random.seed(9)
            dataList = list(range(groupCount))
            insertIndex = []  # 选出嵌入索引
            for i in range(int(groupCount / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            #            print("insertIndex为：",insertIndex)
            functionNameList = splitList(functionNameList, 8)  # 将函数名分为8个一组
            for i, eachFunNameList in enumerate(functionNameList):
                #                print("正在处理第",i+1,"组")
                #                print("第",i+1,"组的函数列表为：",eachFunNameList)
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
                    eachCoverConFeature.append(functionPerm.index(functionNameTuple))  # 载体的函数顺序特征
                    if insertIndex.count(i) == 0:
                        #                        print("不嵌入秘密信息")
                        eachStegoConFeature.append(functionPerm.index(functionNameTuple))
                    else:
                        #                        print("嵌入秘密信息")
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
                        eachStegoConFeature.append(functionPerm.index(tuple(funNameAfterPerm)))
                        #                    print("之前求的顺序特征为：",permIndex)
                        #            stegoFunctionFeature.append(permIndex) #stego的函数顺序特征
                        print("原enMessage的长度为：", len(enMessage))
                        enMessage = enMessage[bitCount:len(enMessage)]
                        print("新enMessage的长度为：", len(enMessage))
            #            print("第",x,"个合约的eachCoverConFeature为：",eachCoverConFeature)
            #            print("第",x,"个合约的eachStegoConFeature为：",eachStegoConFeature)
            strCoverFeature = ''
            strStegoFeature = ''
            for var in eachCoverConFeature:
                strCoverFeature += bin(var)[2:len(bin(var))]
            #            print("strCoverFeature为：",strCoverFeature)
            #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
            for var in eachStegoConFeature:
                strStegoFeature += bin(var)[2:len(bin(var))]
            #            print("steStegoFeature为：",strStegoFeature)
            #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
            coveFunctionFeature.append(int(strCoverFeature, base=2))
            stegoFunctionFeature.append(int(strStegoFeature, base=2))
    #    print("cover的函数顺序特征为：",coveFunctionFeature)
    #    print("coveFunctionFeature的长度为：",coveFunctionFeature)
    #    print("stego的函数顺序特征为：",stegoFunctionFeature)
    #    print("stegoFunctionFeature的长度为：",len(stegoFunctionFeature))
    return coveFunctionFeature, stegoFunctionFeature


def get_fun_Feature():
    coveFunctionFeature = []  # 载体合约中的函数特征列表
    stegoFunctionFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 200):
        eachCoverConFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoConFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchFunction = re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        if len(matchFunction) > 1:  # 匹配到的函数数目大于1
            #            print("匹配到的函数数目为：",len(matchFunction))
            functionNameList = []
            for var in matchFunction:  # 获取函数名列表
                #        print(var)
                split_List = re.split(r'[,\()]', var)
                functionName = split_List[0].replace('function', '').strip()
                functionNameList.append(functionName)  # 获取函数名列表
            #            print("函数名列表的长度为：",len(functionName))
            functionNameList = splitList(functionNameList, 8)  # 将函数名分为8个一组
            for eachFunNameList in functionNameList:
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
                    eachCoverConFeature.append(functionPerm.index(functionNameTuple))  # 载体的函数顺序特征
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
                    eachStegoConFeature.append(functionPerm.index(tuple(funNameAfterPerm)))
                    #                    print("之前求的顺序特征为：",permIndex)
                    #            stegoFunctionFeature.append(permIndex) #stego的函数顺序特征
                    print("原enMessage的长度为：", len(enMessage))
                    enMessage = enMessage[bitCount:len(enMessage)]
                    print("新enMessage的长度为：", len(enMessage))
            #            print("第",x,"个合约的eachCoverConFeature为：",eachCoverConFeature)
            #            print("第",x,"个合约的eachStegoConFeature为：",eachStegoConFeature)
            strCoverFeature = ''
            strStegoFeature = ''
            for var in eachCoverConFeature:
                strCoverFeature += bin(var)[2:len(bin(var))]
            #            print("strCoverFeature为：",strCoverFeature)
            #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
            for var in eachStegoConFeature:
                strStegoFeature += bin(var)[2:len(bin(var))]
            #            print("steStegoFeature为：",strStegoFeature)
            #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
            coveFunctionFeature.append(int(strCoverFeature, base=2))
            stegoFunctionFeature.append(int(strStegoFeature, base=2))
    #    print("cover的函数顺序特征为：",coveFunctionFeature)
    #    print("coveFunctionFeature的长度为：",coveFunctionFeature)
    #    print("stego的函数顺序特征为：",stegoFunctionFeature)
    #    print("stegoFunctionFeature的长度为：",len(stegoFunctionFeature))
    return coveFunctionFeature, stegoFunctionFeature


# 获取合约的0，1分布特征
def get_zero_one_Feature():
    coveFunctionFeature = []  # 载体合约中的函数特征列表
    stegoFunctionFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 20000):
        print("第", x, "个合约：")
        eachCoverConFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoConFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchIter = re.finditer(r'(function|event)\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        matchFunction = []
        for eachFunction in matchIter:
            matchFunction.append(eachFunction.group())
        if len(matchFunction) > 1:  # 匹配到的函数数目大于1
            #            print("匹配到的函数数目为：",len(matchFunction))
            functionNameList = []
            for var in matchFunction:  # 获取函数名列表
                #                print(var)
                split_List = re.split(r'[,\()]', var)
                functionName = split_List[0].replace('function', '').strip()
                functionNameList.append(functionName)  # 获取函数名列表
            #            print("函数名列表的长度为：",len(functionName))
            functionNameList = splitList(functionNameList, 8)  # 将函数名分为8个一组
            for eachFunNameList in functionNameList:
                if len(eachFunNameList) > 1:  # 只处理长度大于1的函数名列表
                    funNameHashList = []  # 函数名哈希值列表
                    hashNameDict = {}  # 创建哈希值和name的对应关系字典
                    for eachFunName in eachFunNameList:
                        funNameHashList.append(hash(eachFunName))
                        hashNameDict[hash(eachFunName)] = eachFunName
                    functionNameTuple = tuple(eachFunNameList)
                    #                    print("原函数名的顺序为：",functionNameTuple)
                    #                    print("原函数名哈希值的顺序为：",functionNameHashTuple)

                    # 获取bitCount
                    functionNumFactorial = math.factorial(len(eachFunNameList))  # 函数数数目的阶乘
                    functionNumLog = math.log(functionNumFactorial) / math.log(2)  # 阶乘的对数
                    # bitCount表示最低嵌入的位数
                    bitCount = int(functionNumLog)

                    # 获取eachCoverConFeature
                    functionPerm = list(itertools.permutations(eachFunNameList))
                    functionPerm.sort()
                    formatControlLow = '{:0' + str(bitCount) + 'b}'
                    formatControlHigh = '{:0' + str(bitCount + 1) + 'b}'
                    coverPermIndex = functionPerm.index(functionNameTuple)
                    # 根据索引求出二进制串
                    if coverPermIndex < (functionNumFactorial - 2 ** bitCount) * 2:  # 嵌入bitCount+1位
                        eachCoverFeature = formatControlHigh.format(coverPermIndex)
                        #                        print("eachCoverFeature为：",eachCoverFeature)
                        eachCoverConFeature.append(eachCoverFeature)
                    else:  # 嵌入bitCount位
                        eachCoverFeature = formatControlLow.format(
                            coverPermIndex - functionNumFactorial + 2 ** bitCount)
                        #                        print("eachCoverFeature为：",eachCoverFeature)
                        eachCoverConFeature.append(eachCoverFeature)

                    # 获取eachStegoConFeature
                    # 根据enMessage判断嵌入位数
                    # 嵌入bitCount+1位
                    if int(enMessage[0:bitCount + 1], base=2) < (functionNumFactorial - 2 ** bitCount) * 2:
                        stegoBitCount = bitCount + 1
                    else:
                        stegoBitCount = bitCount
                    #                    print("eachStegoFeature为：",enMessage[0:stegoBitCount])
                    eachStegoConFeature.append(enMessage[0:stegoBitCount])

                    # 处理enMessage
                    #                    print("原enMessage的长度为：",len(enMessage))
                    enMessage = enMessage[stegoBitCount:len(enMessage)]
            #                    print("新enMessage的长度为：",len(enMessage))
            #            print("第",x,"个合约的eachCoverConFeature为：",eachCoverConFeature)
            #            print("第",x,"个合约的eachStegoConFeature为：",eachStegoConFeature)
            strCoverFeature = ''
            strStegoFeature = ''
            for var in eachCoverConFeature:
                #                print("var",var)
                strCoverFeature += var
            #            print("strCoverFeature为：",strCoverFeature)
            coverZerolen = strCoverFeature.count('0')
            coverOneLen = strCoverFeature.count('1')
            coveFunctionFeature.append((coverZerolen - coverOneLen) / (float(coverZerolen) + coverOneLen))
            for var in eachStegoConFeature:
                strStegoFeature += var
            #            print("steStegoFeature为：",strStegoFeature)
            #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
            stegoZerolen = strStegoFeature.count('0')
            stegoOneLen = strStegoFeature.count('1')
            stegoFunctionFeature.append((stegoZerolen - stegoOneLen) / (float(stegoZerolen) + stegoOneLen))
    #            coveFunctionFeature.append(int(strCoverFeature,base=2))
    #            stegoFunctionFeature.append(int(strStegoFeature,base=2))
    #    print("cover的函数顺序特征为：",coveFunctionFeature)
    #    print("coveFunctionFeature的长度为：",coveFunctionFeature)
    #    print("stego的函数顺序特征为：",stegoFunctionFeature)
    #    print("stegoFunctionFeature的长度为：",len(stegoFunctionFeature))
    return coveFunctionFeature, stegoFunctionFeature


def get_zeroOneRandom_Feature():
    coveFunctionFeature = []  # 载体合约中的函数特征列表
    stegoFunctionFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 20000):
        print("开始处理第", x, "个合约")
        eachCoverFunFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoFunFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchIter = re.finditer(r'(function|event)\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        matchFunction = []
        for eachFunction in matchIter:
            matchFunction.append(eachFunction.group())
        if len(matchFunction) > 1:  # 匹配到的函数数目大于1
            #            print("匹配到的函数数目为：",len(matchFunction))
            functionNameList = []
            for var in matchFunction:  # 获取函数名列表
                #        print(var)
                split_List = re.split(r'[,\()]', var)
                functionName = split_List[0].replace('function', '').strip()
                functionName = functionName.replace('event', '').strip()
                #                print("函数名为：",functionName)
                functionNameList.append(functionName)  # 获取函数名列表
            #            print("函数名列表的长度为：",len(functionNameList))
            funCount = len(functionNameList)
            groupCount = math.ceil(funCount / 8)
            #            print("函数列表可以分为",groupCount,"组")

            # 使用随机数种子决定是否嵌入数据
            random.seed(9)
            dataList = list(range(groupCount))
            insertIndex = []

            # 选出嵌入索引
            for i in range(int(groupCount / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            #            print("insertIndex为：",insertIndex)

            functionNameList = splitList(functionNameList, 8)  # 将函数名分为8个一组
            for i, eachFunNameList in enumerate(functionNameList):
                #                print("正在处理第",i+1,"组")
                #                print("第",i+1,"组的函数列表为：",eachFunNameList)
                if len(eachFunNameList) > 1:  # 只处理长度大于1的函数名列表
                    funNameHashList = []  # 函数名哈希值列表
                    hashNameDict = {}  # 创建哈希值和name的对应关系字典
                    for eachFunName in eachFunNameList:
                        funNameHashList.append(hash(eachFunName))
                        hashNameDict[hash(eachFunName)] = eachFunName
                    functionNameTuple = tuple(eachFunNameList)
                    #                    print("原函数名的顺序为：",functionNameTuple)
                    #                    print("原函数名哈希值的顺序为：",functionNameHashTuple)

                    # 获取bitCount
                    functionNumFactorial = math.factorial(len(eachFunNameList))  # 函数数数目的阶乘
                    functionNumLog = math.log(functionNumFactorial) / math.log(2)  # 阶乘的对数
                    # bitCount表示最低嵌入的位数
                    bitCount = int(functionNumLog)

                    # 获取eachCoverConFeature
                    functionPerm = list(itertools.permutations(eachFunNameList))
                    functionPerm.sort()
                    formatControlLow = '{:0' + str(bitCount) + 'b}'
                    formatControlHigh = '{:0' + str(bitCount + 1) + 'b}'
                    coverPermIndex = functionPerm.index(functionNameTuple)

                    # 根据排列索引求出二进制串
                    if coverPermIndex < (functionNumFactorial - 2 ** bitCount) * 2:  # 嵌入bitCount+1位
                        eachCoverFeature = formatControlHigh.format(coverPermIndex)
                        # print("eachCoverFeature为：", eachCoverFeature)
                        eachCoverFunFeature.append(eachCoverFeature)
                    else:  # 嵌入bitCount位
                        eachCoverFeature = formatControlLow.format(
                            coverPermIndex - functionNumFactorial + 2 ** bitCount)
                        # print("eachCoverFeature为：", eachCoverFeature)
                        eachCoverFunFeature.append(eachCoverFeature)

                    if insertIndex.count(i) == 0:
                        # print("不嵌入秘密信息")
                        # print("eachStegoConFeature值和eachCoverFeature相等：", eachCoverFeature)
                        eachStegoFunFeature.append(eachCoverFeature)
                    else:
                        # print("嵌入秘密信息")
                        if int(enMessage[0:bitCount + 1], base=2) < (functionNumFactorial - 2 ** bitCount) * 2:
                            stegoBitCount = bitCount + 1
                        else:
                            stegoBitCount = bitCount
                        # print("eachStegoFeature为：", enMessage[0:stegoBitCount])
                        eachStegoFunFeature.append(enMessage[0:stegoBitCount])

                        # 处理enMessage
                        #                        print("原enMessage的长度为：",len(enMessage))
                        enMessage = enMessage[stegoBitCount:len(enMessage)]
            #                        print("新enMessage的长度为：",len(enMessage))
            #            print("第",x,"个合约的eachCoverConFeature为：",eachCoverConFeature)
            #            print("第",x,"个合约的eachStegoConFeature为：",eachStegoConFeature)
            strCoverFeature = ''
            strStegoFeature = ''
            for var in eachCoverFunFeature:
                strCoverFeature += var
            #            print("strCoverFeature为：",strCoverFeature)
            #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
            coverZerolen = strCoverFeature.count('0')
            coverOneLen = strCoverFeature.count('1')
            coveFunctionFeature.append((coverZerolen - coverOneLen) / (float(coverZerolen) + coverOneLen))
            for var in eachStegoFunFeature:
                strStegoFeature += var
            #            print("steStegoFeature为：",strStegoFeature)
            #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
            stegoZerolen = strStegoFeature.count('0')
            stegoOneLen = strStegoFeature.count('1')
            stegoFunctionFeature.append((stegoZerolen - stegoOneLen) / (float(stegoZerolen) + stegoOneLen))
    #    print("cover的函数顺序特征为：",coveFunctionFeature)
    #    print("coveFunctionFeature的长度为：",coveFunctionFeature)
    #    print("stego的函数顺序特征为：",stegoFunctionFeature)
    #    print("stegoFunctionFeature的长度为：",len(stegoFunctionFeature))
    return coveFunctionFeature, stegoFunctionFeature


def train():
    #    coveFunctionFeature,stegoFunctionFeature=get_fun_Feature()
    #    coveFunctionFeature,stegoFunctionFeature=get_funRandom_Feature()
    # coveFunctionFeature, stegoFunctionFeature = get_zero_one_Feature()
    coveFunctionFeature, stegoFunctionFeature = get_zeroOneRandom_Feature()
    X = coveFunctionFeature + stegoFunctionFeature  # 获取所有的特征
    #    print("所有的特征为：",X)
    print("所有的特征长度为：", len(X))
    X = np.array(X).reshape(-1, 1)
    #    print("X为：",X)

    # 标准化
    #    scaler=StandardScaler().fit(X)
    #    X_std=scaler.transform(X)

    # 归一化到[0,1]
    #    min_max_scaler = preprocessing.MinMaxScaler()
    #    X_minMax=min_max_scaler.fit_transform(X)

    # 归一化到[-1,1]
    #    max_abs_scaler = preprocessing.MaxAbsScaler()
    #    X_train_maxsbs = max_abs_scaler.fit_transform(X)
    #    print("X_train_maxsbs为：",X_train_maxsbs)

    y = [0] * len(coveFunctionFeature) + [1] * len(stegoFunctionFeature)  # 获取所有的标签
    #    y=np.array(y).reshape(-1,1)
    #    print("所有的标签为：",y)
    print("所有的标签长度为：", len(y))
    train_X, test_X, train_y, test_true_y = train_test_split(X, y, test_size=0.3, random_state=0)

    # 线性核预测
    #    svm_model_linear=svm.SVC(kernel='linear',C=100.) #线性核
    #    svm_model_linear.fit(train_X,train_y)
    #    test_predict_linear_y=svm_model_linear.predict(test_X) #预测值
    #    print("预测值为：",test_predict_linear_y)
    #    print("SVM线性核预测的准确度为：",accuracy_score(test_true_y,test_predict_linear_y))

    # 高斯核预测
    #    svm_model_rbf=svm.SVC(kernel='rbf',gamma=0.44,C=52.2) #100%容量时的最优参数
    svm_model_rbf = svm.SVC(kernel='rbf', gamma=0.64, C=25.5)  # 50%时的最优参数
    svm_model_rbf.fit(train_X, train_y)
    #    print("test_X为：",test_X)
    test_predict_rbf_y = svm_model_rbf.predict(test_X)  # 预测值
    print("预测值为：", test_predict_rbf_y)
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))
    con_matrix = confusion_matrix(y_true=test_true_y, y_pred=test_predict_rbf_y)
    print("混淆矩阵为：", con_matrix)
    print("FP rate:", 100 * con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1]), "%")
    print("FN rate:", 100 * con_matrix[0][1] / (con_matrix[0][0] + con_matrix[0][1]), "%")
    drawHist(coveFunctionFeature, stegoFunctionFeature)


# rbf核预测，网格搜索最优参数
#    model=svm.SVC(kernel='rbf')
#    c_can=np.linspace(20,30,10)
#    gamma_can = np.linspace(0.6, 0.7, 10)
#    svc_rbf = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
#    svc_rbf.fit(train_X,train_y)
#    print ('最优参数：\n', svc_rbf.best_params_)

# 数据可视化
#    train_data=np.concatenate((train_X,train_y),axis = 1)
#    print("train_X为：",train_X,"长度为：",len(train_X))
#    print("train_y为：",train_y,"长度为：",len(train_y))
##    print("train_data为：",train_data)
##    print(train_data[:,0],train_data[:,0].shape,type(train_data[:,0]))
##    print(train_data[:,1],train_data[:,1].shape)
#    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#    ax = plt.subplot()
#    ax.set_title("Input data")
#    # Plot the training points
#    ax.scatter(train_X, train_y, c=train_y, cmap=cm_bright)
#    ax.set_xticks(())
#    ax.set_yticks(())
#    plt.tight_layout()
#    plt.show()


if __name__ == '__main__':
    train()
#    coveFunctionFeature,stegoFunctionFeature=get_funRandom_Feature()
#    print("coveFunctionFeature为",coveFunctionFeature)
#    print("coveFunctionFeature的长度为",len(coveFunctionFeature))
#    print("stegoFunctionFeature为：",stegoFunctionFeature)
#    print("stegoFunctionFeature的长度为：",len(stegoFunctionFeature))
