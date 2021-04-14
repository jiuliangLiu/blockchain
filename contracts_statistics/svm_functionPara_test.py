# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:30:51 2019
用SVM检测函数参数排序
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
import random
import matplotlib.pyplot as plt
from svm_uint_test import drawHist

base_read_path = r'F:\experimentData\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path = r'F:\experimentData\message1.txt'  # 秘密信息存储路径


def get_funParaRandom_feature():
    coveFunctionParaFeature = []  # 载体合约中的函数特征列表
    stegoFunctionParaFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 20000):
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchIter = re.finditer(r'(function|event)\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        matchFunction = []
        for eachFunction in matchIter:
            matchFunction.append(eachFunction)
        #        print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
        # 如果函数个数大于0，此时一定可以匹配到函数参数
        if (len(matchFunction) > 0):
            eachCoverFuncParaFeature = []  # 每个Cover合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值
            eachStegoFuncParaFeature = []  # 每个Stego合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值

            # 使用随机数种子决定是否嵌入数据
            random.seed(9)
            dataList = list(range(len(matchFunction)))
            insertIndex = []  # 选出嵌入索引
            for i in range(int(len(matchFunction) / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            print("insertIndex为：", insertIndex)
            for funIndex, eachFun in enumerate(matchFunction):
                if funIndex > 8:
                    break
                splitList = re.split(r'[,\()]', eachFun)  # 分隔匹配到的函数
                paraList = []  # 获取每个函数的参数列表
                paraHashList = []  # 获取每个函数参数的哈希值列表
                hashParaDict = {}  # 函数参数和哈希值的对应字典
                for funParaIndex, funPara in enumerate(splitList):
                    #            print(funPara)
                    if funParaIndex != 0 and funParaIndex != len(splitList) - 1:
                        #                print(funPara)
                        paraList.append(funPara.strip())
                        paraHashList.append(hash(funPara.strip()))
                        hashParaDict[hash(funPara.strip())] = funPara.strip()
                #                print("某个函数的参数列表为：",paraList)
                #                print("某个函数参数个数为：",len(paraList))
                #                print("某个函数参数的哈希值列表为：",paraHashList)
                if 1 < len(paraList) < 10:  # 如果参数大于等于2个
                    paraTuple = tuple(paraList)
                    #                    print(paraTuple)
                    paraPerm = list(itertools.permutations(paraList))
                    paraPerm.sort()  # 排列之后进行排序
                    paraHashPerm = list(itertools.permutations(paraHashList))
                    paraHashPerm.sort()
                    #            print(paraPerm)
                    #                    print('函数参数的顺序特征值为：',paraPerm.index(paraTuple))
                    eachCoverFuncParaFeature.append(paraPerm.index(paraTuple))

                    print("索引值为：", funIndex)
                    if insertIndex.count(funIndex) == 0:
                        print("不嵌入秘密信息")
                        eachStegoFuncParaFeature.append(paraPerm.index(paraTuple))
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
                        eachStegoFuncParaFeature.append(paraPerm.index(tuple(paraAfterPerm)))
                        print("原enMessage的长度为：", len(enMessage))
                        enMessage = enMessage[bitCount:len(enMessage)]
                        print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverFuncParaFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverFuncParaFeature:
                    strCoverFeature += bin(var)[2:len(bin(var))]
                #                print("strCoverFeature为：",strCoverFeature)
                #                print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoFuncParaFeature:
                    strStegoFeature += bin(var)[2:len(bin(var))]
                #                print("strStegoFeature为：",strStegoFeature)
                #                print("eachStegoFeature为：",int(strStegoFeature,base=2))
                coveFunctionParaFeature.append(int(strCoverFeature, base=2))
                stegoFunctionParaFeature.append(int(strStegoFeature, base=2))
            #    print("所有cover合约的函数参数特征列表为：",coveFunctionParaFeature)
    #    print("所有cover合约的函数参数特征列表长度为：",len(coveFunctionParaFeature))
    #    print("所有stego合约的函数参数特征列表为：",coveFunctionParaFeature)
    #    print("所有stego合约的函数参数特征列表长度为：",len(coveFunctionParaFeature))
    return coveFunctionParaFeature, stegoFunctionParaFeature


def get_funPara_feature():
    coveFunctionParaFeature = []  # 载体合约中的函数特征列表
    stegoFunctionParaFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 200):
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchFunction = re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        #        print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
        # 如果函数个数大于0，此时一定可以匹配到函数参数
        if (len(matchFunction) > 0):
            eachCoverFuncParaFeature = []  # 每个Cover合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值
            eachStegoFuncParaFeature = []  # 每个Stego合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值
            for index, eachFun in enumerate(matchFunction):
                if index > 8:
                    break
                splitList = re.split(r'[,\()]', eachFun)  # 分隔匹配到的函数
                paraList = []  # 获取每个函数的参数列表
                paraHashList = []  # 获取每个函数参数的哈希值列表
                hashParaDict = {}  # 函数参数和哈希值的对应字典
                for index, funPara in enumerate(splitList):
                    #            print(funPara)
                    if index != 0 and index != len(splitList) - 1:
                        #                print(funPara)
                        paraList.append(funPara.strip())
                        paraHashList.append(hash(funPara.strip()))
                        hashParaDict[hash(funPara.strip())] = funPara.strip()
                #                print("某个函数的参数列表为：",paraList)
                #                print("某个函数参数个数为：",len(paraList))
                #                print("某个函数参数的哈希值列表为：",paraHashList)
                if 1 < len(paraList) < 10:  # 如果参数大于等于2个
                    paraTuple = tuple(paraList)
                    #                    print(paraTuple)
                    paraPerm = list(itertools.permutations(paraList))
                    paraPerm.sort()  # 排列之后进行排序
                    paraHashPerm = list(itertools.permutations(paraHashList))
                    paraHashPerm.sort()
                    #            print(paraPerm)
                    #                    print('函数参数的顺序特征值为：',paraPerm.index(paraTuple))
                    eachCoverFuncParaFeature.append(paraPerm.index(paraTuple))
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
                    eachStegoFuncParaFeature.append(paraPerm.index(tuple(paraAfterPerm)))
                    print("原enMessage的长度为：", len(enMessage))
                    enMessage = enMessage[bitCount:len(enMessage)]
                    print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverFuncParaFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverFuncParaFeature:
                    strCoverFeature += bin(var)[2:len(bin(var))]
                #                print("strCoverFeature为：",strCoverFeature)
                #                print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoFuncParaFeature:
                    strStegoFeature += bin(var)[2:len(bin(var))]
                #                print("strStegoFeature为：",strStegoFeature)
                #                print("eachStegoFeature为：",int(strStegoFeature,base=2))
                coveFunctionParaFeature.append(int(strCoverFeature, base=2))
                stegoFunctionParaFeature.append(int(strStegoFeature, base=2))
            #    print("所有cover合约的函数参数特征列表为：",coveFunctionParaFeature)
    #    print("所有cover合约的函数参数特征列表长度为：",len(coveFunctionParaFeature))
    #    print("所有stego合约的函数参数特征列表为：",coveFunctionParaFeature)
    #    print("所有stego合约的函数参数特征列表长度为：",len(coveFunctionParaFeature))
    return coveFunctionParaFeature, stegoFunctionParaFeature


def get_funPara_zeroOne_feature():
    coveFunctionParaFeature = []  # 载体合约中的函数特征列表
    stegoFunctionParaFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 20000):
        print("正在处理第", x, "个合约")
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchIter = re.finditer(r'(function|event)\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        matchFunction = []
        for eachFunction in matchIter:
            matchFunction.append(eachFunction.group())
        #        print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
        # 如果函数个数大于0，此时一定可以匹配到函数参数
        if (len(matchFunction) > 0):
            eachCoverFunParaFeature = []  # 每个Cover合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值
            eachStegoFunParaFeature = []  # 每个Stego合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值
            for index, eachFun in enumerate(matchFunction):
                #                if index>8:
                #                    break
                splitList = re.split(r'[,\()]', eachFun)  # 分隔匹配到的函数
                paraList = []  # 获取每个函数的参数列表
                paraHashList = []  # 获取每个函数参数的哈希值列表
                hashParaDict = {}  # 函数参数和哈希值的对应字典
                for index, funPara in enumerate(splitList):
                    #            print(funPara)
                    if index != 0 and index != len(splitList) - 1:
                        #                print(funPara)
                        paraList.append(funPara.strip())
                        paraHashList.append(hash(funPara.strip()))
                        hashParaDict[hash(funPara.strip())] = funPara.strip()
                #                print("某个函数的参数列表为：",paraList)
                #                print("某个函数参数个数为：",len(paraList))
                #                print("某个函数参数的哈希值列表为：",paraHashList)
                if 1 < len(paraList) < 10:  # 如果参数大于等于2个
                    # 获取bitCount和permIndex
                    varFactorial = math.factorial(len(paraList))  # 函数参数数目的阶乘
                    varlog = math.log(varFactorial) / math.log(2)  # 阶乘的对数
                    # bitCount表示最低嵌入的位数
                    bitCount = int(varlog)

                    # 获取eachCoverFuncParaFeature
                    paraTuple = tuple(paraList)
                    #                    print(paraTuple)
                    paraPerm = list(itertools.permutations(paraList))
                    paraPerm.sort()  # 排列之后进行排序
                    formatControlLow = '{:0' + str(bitCount) + 'b}'
                    formatControlHigh = '{:0' + str(bitCount + 1) + 'b}'
                    coverPermIndex = paraPerm.index(paraTuple)
                    if coverPermIndex < (varFactorial - 2 ** bitCount) * 2:  # 嵌入bitCount+1位
                        eachCoverFeature = formatControlHigh.format(coverPermIndex)
                        #                        print("eachCoverFeature为：",eachCoverFeature)
                        eachCoverFunParaFeature.append(eachCoverFeature)
                    else:  # 嵌入bitCount位
                        eachCoverFeature = formatControlLow.format(coverPermIndex - varFactorial + 2 ** bitCount)
                        #                        print("eachCoverFeature为：",eachCoverFeature)
                        eachCoverFunParaFeature.append(eachCoverFeature)

                    # 获取eachStegoFuncParaFeature
                    # 根据enMessage判断嵌入位数
                    # 嵌入bitCount+1位
                    if int(enMessage[0:bitCount + 1], base=2) < (varFactorial - 2 ** bitCount) * 2:
                        stegoBitCount = bitCount + 1
                    else:
                        stegoBitCount = bitCount
                    eachStegoFunParaFeature.append(enMessage[0:stegoBitCount])
                    # 处理enMessage
                    # print("原enMessage的长度为：", len(enMessage))
                    enMessage = enMessage[stegoBitCount:len(enMessage)]
                    # print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverFunParaFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverFunParaFeature:
                    #                    print("var",var)
                    strCoverFeature += var
                # print("strCoverFeature", strCoverFeature)
                coverZerolen = strCoverFeature.count('0')
                coverOneLen = strCoverFeature.count('1')
                coveFunctionParaFeature.append((coverZerolen - coverOneLen) / (float(coverZerolen) + coverOneLen))

                for var in eachStegoFunParaFeature:
                    strStegoFeature += var
                # print("strStegoFeature", strStegoFeature)
                stegoZerolen = strStegoFeature.count('0')
                stegoOneLen = strStegoFeature.count('1')
                stegoFunctionParaFeature.append((stegoZerolen - stegoOneLen) / (float(stegoZerolen) + stegoOneLen))
    return coveFunctionParaFeature, stegoFunctionParaFeature


def get_funPara_zeroOneRandom_feature():
    coveFunctionParaFeature = []  # 载体合约中的函数特征列表
    stegoFunctionParaFeature = []  # 含秘载体合约中的函数特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 20000):
        print("开始处理第", x, "个合约")
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchIter = re.finditer(r'(function|event)\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)', source_code)
        matchFunction = []
        for eachFunction in matchIter:
            matchFunction.append(eachFunction.group())
        #        print('第',x,'个合约匹配到的函数总数为：',len(matchFunction))
        # 如果函数个数大于0，此时一定可以匹配到函数参数
        if (len(matchFunction) > 0):
            eachCoverFunParaFeature = []  # 每个Cover合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值
            eachStegoFunParaFeature = []  # 每个Stego合约中的所有函数的特征值列表，最终需要将此列表转为一个特征值

            # 使用随机数种子决定是否嵌入数据
            random.seed(9)
            dataList = list(range(len(matchFunction)))
            insertIndex = []
            # 选出嵌入索引
            for i in range(int(len(matchFunction) / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            # print("insertIndex为：", insertIndex)

            for funIndex, eachFun in enumerate(matchFunction):
                #                if funIndex>8:
                #                    break
                splitList = re.split(r'[,\()]', eachFun)  # 分隔匹配到的函数
                paraList = []  # 获取每个函数的参数列表
                paraHashList = []  # 获取每个函数参数的哈希值列表
                hashParaDict = {}  # 函数参数和哈希值的对应字典
                for funParaIndex, funPara in enumerate(splitList):
                    #            print(funPara)
                    if funParaIndex != 0 and funParaIndex != len(splitList) - 1:
                        #                print(funPara)
                        paraList.append(funPara.strip())
                        paraHashList.append(hash(funPara.strip()))
                        hashParaDict[hash(funPara.strip())] = funPara.strip()
                #                print("某个函数的参数列表为：",paraList)
                #                print("某个函数参数个数为：",len(paraList))
                #                print("某个函数参数的哈希值列表为：",paraHashList)
                if 1 < len(paraList) < 10:  # 如果参数大于等于2个

                    # 获取bitCount
                    varFactorial = math.factorial(len(paraList))  # 函数参数数目的阶乘
                    varlog = math.log(varFactorial) / math.log(2)  # 阶乘的对数
                    # bitCount表示最低嵌入的位数
                    bitCount = int(varlog)

                    # 获取eachCoverFuncParaFeature
                    paraTuple = tuple(paraList)
                    #                    print(paraTuple)
                    paraPerm = list(itertools.permutations(paraList))
                    paraPerm.sort()  # 排列之后进行排序
                    formatControlLow = '{:0' + str(bitCount) + 'b}'
                    formatControlHigh = '{:0' + str(bitCount + 1) + 'b}'
                    coverPermIndex = paraPerm.index(paraTuple)
                    # 根据排列索引求出二进制串
                    if coverPermIndex < (varFactorial - 2 ** bitCount) * 2:  # 嵌入bitCount+1位
                        eachCoverFeature = formatControlHigh.format(coverPermIndex)
                        # print("eachCoverFeature为：", eachCoverFeature)
                        eachCoverFunParaFeature.append(eachCoverFeature)
                    else:  # 嵌入bitCount位
                        eachCoverFeature = formatControlLow.format(coverPermIndex - varFactorial + 2 ** bitCount)
                        # print("eachCoverFeature为：", eachCoverFeature)
                        eachCoverFunParaFeature.append(eachCoverFeature)

                        # 获取eachStegoFuncParaFeature
                    if insertIndex.count(funIndex) == 0:
                        # print("不嵌入秘密信息")
                        # print("eachStegoFuncParaFeature的值为：", eachCoverFeature)
                        eachStegoFunParaFeature.append(eachCoverFeature)
                    else:
                        # print("嵌入秘密信息")
                        if int(enMessage[0:bitCount + 1], base=2) < (varFactorial - 2 ** bitCount) * 2:
                            stegoBitCount = bitCount + 1
                        else:
                            stegoBitCount = bitCount
                        # print("eachStegoFeature为：", enMessage[0:stegoBitCount])
                        eachStegoFunParaFeature.append(enMessage[0:stegoBitCount])
                        # 处理enMessage
                        # print("原enMessage的长度为：", len(enMessage))
                        # enMessage = enMessage[stegoBitCount:len(enMessage)]
                        # print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverFunParaFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverFunParaFeature:
                    strCoverFeature += var
                #                print("strCoverFeature为：",strCoverFeature)
                coverZerolen = strCoverFeature.count('0')
                coverOneLen = strCoverFeature.count('1')
                coveFunctionParaFeature.append((coverZerolen - coverOneLen) / (float(coverZerolen) + coverOneLen))
                for var in eachStegoFunParaFeature:
                    strStegoFeature += var
                #                print("strStegoFeature为：",strStegoFeature)
                stegoZerolen = strStegoFeature.count('0')
                stegoOneLen = strStegoFeature.count('1')
                stegoFunctionParaFeature.append((stegoZerolen - stegoOneLen) / (float(stegoZerolen) + stegoOneLen))
            #    print("所有cover合约的函数参数特征列表为：",coveFunctionParaFeature)
    #    print("所有cover合约的函数参数特征列表长度为：",len(coveFunctionParaFeature))
    #    print("所有stego合约的函数参数特征列表为：",coveFunctionParaFeature)
    #    print("所有stego合约的函数参数特征列表长度为：",len(coveFunctionParaFeature))
    return coveFunctionParaFeature, stegoFunctionParaFeature


def train():
    #    coveFunctionParaFeature,stegoFunctionParaFeature=get_funPara_feature()
    #    coveFunctionParaFeature,stegoFunctionParaFeature=get_funParaRandom_feature()
    # coveFunctionParaFeature, stegoFunctionParaFeature = get_funPara_zeroOne_feature()
    coveFunctionParaFeature, stegoFunctionParaFeature = get_funPara_zeroOneRandom_feature()
    X = coveFunctionParaFeature + stegoFunctionParaFeature
    X = np.array(X).reshape(-1, 1)
    print("总特征长度为：", len(X))
    lenCover = int(len(coveFunctionParaFeature))
    lenStego = int(len(stegoFunctionParaFeature))
    print("Cover特征数组的长度为：", lenCover)
    print("Stego特征数组的长度为", lenStego)
    y = [0] * lenCover + [1] * lenStego
    print("所有的标签长度为：", len(y))
    train_X, test_X, train_y, test_true_y = train_test_split(X, y, test_size=0.3, random_state=0)
    #    train_X=np.array(train_X).reshape(-1,1)
    #    print('训练集的特征为：',train_X)
    print('训练集的特征长度为：', len(train_X))
    #    test_X=np.array(test_X).reshape(-1,1)
    #    print('测试集的特征为：',test_X)
    print('测试集的特征长度为：', len(test_X))
    svm_model_rbf = svm.SVC(kernel='rbf', gamma=0.235, C=109.)  # 高斯核
    svm_model_rbf.fit(train_X, train_y)
    test_predict_rbf_y = svm_model_rbf.predict(test_X)  # 预测值
    print("预测值为：", test_predict_rbf_y)
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))
    con_matrix = confusion_matrix(y_true=test_true_y, y_pred=test_predict_rbf_y)
    print("混淆矩阵为：", con_matrix)
    print("FP rate:", 100 * con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1]), "%")
    print("FN rate:", 100 * con_matrix[0][1] / (con_matrix[0][0] + con_matrix[0][1]), "%")
    drawHist(coveFunctionParaFeature, stegoFunctionParaFeature)


if __name__ == '__main__':
    train()
#    coveFunctionParaFeature,stegoFunctionParaFeature=get_new_feature()
#    print("cover特征的长度为：",len(coveFunctionParaFeature))
#    print("stego特征的长度为：",len(stegoFunctionParaFeature))
#    get_funParaRandom_feature()
