# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:22:49 2019
用SVM检测基本块排序
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

base_read_path = r'F:\experimentData\binRuntime\binRuntime{}.bin-runtime'
base_message_path = r'F:\experimentData\message1.txt'  # 秘密信息存储路径
KEY_HASH = 'I am a hash key'  # 哈希密钥
PUSHInst = ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '6a', '6b', '6c', '6d', '6e', '6f', '70', '71',
            '72', '73', '74', '75', '76', '77', '78', '79', '7a', '7b', '7c', '7d', '7e', '7f']
splitInstList = ['56', '57', '5b', '00', 'f3', 'fd']
splitGhiList = ['G', 'H', 'I', 'J', 'K', 'L']
splitInstDict2 = {'G': '56', 'H': '57', 'I': '5b', 'J': '00', 'K': 'f3', 'L': 'fd'}
splitInstDict = {'56': 'G', '57': 'H', '5b': 'I', '00': 'J', 'f3': 'K', 'fd': 'L'}


# 分隔列表
def splitList(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def get_PC_Inst_GhiDict(binCode):
    binCodeDic = {}
    decPC = 0
    # 以下循环用于生成块头字典
    while decPC < len(binCode):
        if PUSHInst.count(binCode[decPC:decPC + 2]) == 1:  # 为PUSH指令
            eachPushInst = binCode[decPC:decPC + 2]
            #                print("PUSHInst为：",eachPushInst)
            #                print("PUSHInst对应的十进制数为：",int(eachPushInst,16))
            operandBits = (int(eachPushInst, 16) - 95) * 2
            #                print("PUSHInst对应的操作数位数为：",operandBits)
            binCodeDic[int(decPC / 2)] = binCode[decPC:decPC + 2 + operandBits]
            #                print("eachCoverBinDic为：",eachCoverBinDic)
            decPC = decPC + 2 + operandBits
        #                print("新的PC值为：",decPC)
        else:  # 指令不为Push指令
            if splitInstList.count(binCode[decPC:decPC + 2]) == 0:  # 不为分割指令
                binCodeDic[int(decPC / 2)] = binCode[decPC:decPC + 2]
            else:
                binCodeDic[int(decPC / 2)] = splitInstDict[binCode[decPC:decPC + 2]]
            decPC = decPC + 2
    return binCodeDic


def get_key_feature():  # 假设敌手无法知道哈希密钥
    coverBBlockFeature = []  # cover基本块的顺序特征列表
    stegoBBlockFeature = []  # stego基本块的顺序特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 500):
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
            binRunCode = fp.read()
        #        print("第",x,"个binRuntime文件内容为:",binRunCode)
        if not binRunCode.strip():
            print("第", x, "个binRuntime文件为空")
        if binRunCode.strip():  # 如果匹配到的binRuntime文件不为空
            binRunCode = binRunCode.replace('56', 'x')
            binRunCode = binRunCode.replace('57', 'y')
            binRunCode = binRunCode.replace('5b', 'z')
            #        print("新字符串为：",binRunCode)
            basicBlock = re.findall(r'[xyz][^xyz]+[xyz]', binRunCode)  # 匹配基本块
            #        print("匹配到的基本块为：",basicBlock)
            #        print("匹配到的基本块数目为：",len(basicBlock))
            basicBlock = splitList(basicBlock, 8)  # 将基本块分为8个一组
            for i, newBBlock in enumerate(basicBlock):  # 对每组基本块进行处理
                #            print("分隔后的基本块列表为：",newBBlock)
                if i > 5:
                    print("i大于8，跳出循环！")
                    break
                if len(newBBlock) > 1:  # 只处理长度大于1的基本块列表
                    bBlockHashList = []  # 基本块哈希列表
                    bBlockHash_keyList = []  # 基本块哈希值列表（带密钥）
                    hashBlockDict = {}  # 创建哈希值和基本块的对应关系字典
                    hash_keyBlockDict = {}  # 创建带密钥哈希值和基本块的对应关系字典
                    for eachBlock in newBBlock:
                        bBlockHashList.append(hash(eachBlock))
                        hashBlockDict[hash(eachBlock)] = eachBlock
                    for eachBlock in newBBlock:
                        bBlockHash_keyList.append(hash(eachBlock + KEY_HASH))
                        hash_keyBlockDict[hash(eachBlock + KEY_HASH)] = eachBlock
                    #            print("哈希字典为：",hashBlockDict)
                    #            print("分隔后的基本块的哈希值列表为：",bBlockHashList)
                    bBlockHashListTuple = tuple(bBlockHashList)  # 原基本块哈希值列表
                    #            print("原哈希基本块的顺序为：",bBlockHashListTuple)
                    bBlockPerm = list(itertools.permutations(bBlockHashList))  # 对基本块的哈希值列表排列
                    bBlock_keyPerm = list(itertools.permutations(bBlockHash_keyList))  # 对基本块的带密钥哈希值列表排列
                    #            bBlockPerm=list(itertools.permutations(newBBlock)) #对基本块列表排列
                    bBlockPerm.sort()  # 字典排序
                    bBlock_keyPerm.sort()
                    #                    print("cover的基本块顺序特征为：",bBlockPerm.index(bBlockHashListTuple)) #敌手不知道哈希密钥，值求哈希值
                    coverBBlockFeature.append(bBlockPerm.index(bBlockHashListTuple))  # 载体的基本块顺序特征
                    blockNumFactorial = math.factorial(len(newBBlock))  # 基本块数目的阶乘
                    #                    print("基本块的数目为：",len(newBBlock))
                    blockNumLog = math.log(blockNumFactorial) / math.log(2)  # 阶乘的对数
                    bitCount = math.floor(blockNumLog)  # 可以存储的位数
                    #                    print("可以存储的位数为：",bitCount)
                    permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                    #                    print("排列索引为：",permIndex)
                    blockHash_keyAfterPerm = list(bBlock_keyPerm[permIndex])  # 排列之后的基本块哈希值列表
                    #            print("排列后的基本块的哈希值顺序为：",blockHashAfterPerm)
                    blockAfterPerm = []  # 排列之后的基本块列表
                    for blockHash in blockHash_keyAfterPerm:
                        blockAfterPerm.append(hash_keyBlockDict[blockHash])
                    #                    print("对应的原基本块列表为:",blockAfterPerm)
                    blockHashAfterPerm = []  # 敌手开始求基本块的哈希值
                    for eachBlock in blockAfterPerm:
                        blockHashAfterPerm.append(hash(eachBlock))
                    print("stego的基本块的顺序特征为:", bBlockPerm.index(tuple(blockHashAfterPerm)))  # 敌手获取的stego顺序特征
                    stegoBBlockFeature.append(bBlockPerm.index(tuple(blockHashAfterPerm)))
                    print("原enMessage的长度为：", len(enMessage))
                    enMessage = enMessage[bitCount:len(enMessage)]
                    print("新enMessage的长度为：", len(enMessage))
    return coverBBlockFeature, stegoBBlockFeature


def get_block_feature():  # 获取每个bin-runtime文件的基本块顺序特征
    coverBBlockFeature = []  # cover基本块的顺序特征列表
    stegoBBlockFeature = []  # stego基本块的顺序特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 200):
        print("开始处理第", x, "个binRuntime")
        eachCoverBinFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoBinFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
            binRunCode = fp.read()
        #        print("第",x,"个binRuntime文件内容为:",binRunCode)
        if not binRunCode.strip():
            print("第", x, "个binRuntime文件为空")
        if binRunCode.strip():  # 如果匹配到的binRuntime文件不为空
            binRunCode = binRunCode.replace('56', 'x')
            binRunCode = binRunCode.replace('57', 'y')
            binRunCode = binRunCode.replace('5b', 'z')
            #        print("新字符串为：",binRunCode)
            basicBlock = re.findall(r'[xyz][^xyz]+[xyz]', binRunCode)  # 匹配基本块
            #        print("匹配到的基本块为：",basicBlock)
            #        print("匹配到的基本块数目为：",len(basicBlock))
            basicBlock = splitList(basicBlock, 8)  # 将基本块分为8个一组
            for i, newBBlock in enumerate(basicBlock):  # 对每组基本块进行处理
                #            print("分隔后的基本块列表为：",newBBlock)
                if i > 8:
                    # print("i>8,跳出循环")
                    break
                if len(newBBlock) > 1:  # 只处理长度大于1的基本块列表
                    bBlockHashList = []  # 基本块哈希列表
                    hashBlockDict = {}  # 创建哈希值和基本块的对应关系字典
                    for eachBlock in newBBlock:
                        bBlockHashList.append(hash(eachBlock))
                        hashBlockDict[hash(eachBlock)] = eachBlock
                    #            print("哈希字典为：",hashBlockDict)
                    #            print("分隔后的基本块的哈希值列表为：",bBlockHashList)
                    bBlockHashListTuple = tuple(bBlockHashList)  # 原基本块哈希值列表
                    #            print("原哈希基本块的顺序为：",bBlockHashListTuple)
                    bBlockPerm = list(itertools.permutations(bBlockHashList))  # 对基本块的哈希值列表排列
                    #            bBlockPerm=list(itertools.permutations(newBBlock)) #对基本块列表排列
                    bBlockPerm.sort()  # 字典排序
                    # print("cover的基本块顺序特征为：", bBlockPerm.index(bBlockHashListTuple))
                    eachCoverBinFeature.append(bBlockPerm.index(bBlockHashListTuple))  # 载体的基本块顺序特征
                    blockNumFactorial = math.factorial(len(newBBlock))  # 基本块数目的阶乘
                    #                    print("基本块的数目为：",len(newBBlock))
                    blockNumLog = math.log(blockNumFactorial) / math.log(2)  # 阶乘的对数
                    #                print(varlog)
                    if blockNumLog - int(blockNumLog) == 0:
                        bitCount = int(blockNumLog)  # 可以存储的位数刚好为整数
                        permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                        # print("可以存储1位", "位数：", bitCount)
                        # print("排列的索引为：", permIndex)
                    else:  # 可以存储的位数不为整数，需要判断存储的位数
                        if int(enMessage[0:int(blockNumLog) + 1], base=2) > (2 ** int(blockNumLog) - 1):
                            bitCount = int(blockNumLog)
                            permIndex = int(enMessage[0:bitCount], base=2) + blockNumFactorial - 2 ** int(
                                blockNumLog)  # 排列索引
                        #                            print("可以存储的位数不为整数,且不能多存一位,存储的位数为：",bitCount)
                        #                            print("排列索引为：",permIndex)
                        else:
                            bitCount = int(blockNumLog) + 1
                            permIndex = int(enMessage[0:bitCount], base=2)
                    blockHashAfterPerm = list(bBlockPerm[permIndex])  # 排列之后的基本块哈希值列表
                    #            print("排列后的基本块的哈希值顺序为：",blockHashAfterPerm)
                    blockAfterPerm = []  # 排列之后的基本块列表
                    for blockHash in blockHashAfterPerm:
                        blockAfterPerm.append(hashBlockDict[blockHash])
                    #            print("对应的原基本块为:",blockAfterPerm)
                    # print("stego的基本块的顺序特征为:", bBlockPerm.index(tuple(blockHashAfterPerm)))
                    eachStegoBinFeature.append(bBlockPerm.index(tuple(blockHashAfterPerm)))
                    # print("原enMessage的长度为：", len(enMessage))
                    enMessage = enMessage[bitCount:len(enMessage)]
                    # print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverBinFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverBinFeature:
                    strCoverFeature += bin(var)[2:len(bin(var))]
                #            print("strCoverFeature为：",strCoverFeature)
                #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoBinFeature:
                    strStegoFeature += bin(var)[2:len(bin(var))]
                #            print("steStegoFeature为：",strStegoFeature)
                #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
                coverBBlockFeature.append(int(strCoverFeature, base=2))
                stegoBBlockFeature.append(int(strStegoFeature, base=2))
    return coverBBlockFeature, stegoBBlockFeature


def get_blockRandom_feature():
    coverBBlockFeature = []  # cover基本块的顺序特征列表
    stegoBBlockFeature = []  # stego基本块的顺序特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 200):
        eachCoverBinFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoBinFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
            binRunCode = fp.read()
        #        print("第",x,"个binRuntime文件内容为:",binRunCode)
        if not binRunCode.strip():
            print("第", x, "个binRuntime文件为空")
        if binRunCode.strip():  # 如果匹配到的binRuntime文件不为空
            binRunCode = binRunCode.replace('56', 'x')
            binRunCode = binRunCode.replace('57', 'y')
            binRunCode = binRunCode.replace('5b', 'z')
            #        print("新字符串为：",binRunCode)
            basicBlock = re.findall(r'[xyz][^xyz]+[xyz]', binRunCode)  # 匹配基本块
            #        print("匹配到的基本块为：",basicBlock)
            #        print("匹配到的基本块数目为：",len(basicBlock))
            blockCount = len(basicBlock)
            groupCount = math.ceil(blockCount / 8)
            print("基本块列表可以分为", groupCount, "组")
            # 使用随机数种子决定是否嵌入数据
            random.seed(9)
            dataList = list(range(groupCount))
            insertIndex = []  # 选出嵌入索引
            for i in range(int(groupCount / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            print("insertIndex为：", insertIndex)
            basicBlock = splitList(basicBlock, 8)  # 将基本块分为8个一组
            for groupIndex, newBBlock in enumerate(basicBlock):  # 对每组基本块进行处理
                #            print("分隔后的基本块列表为：",newBBlock)
                print("groupIndex为：", groupIndex)
                if groupIndex > 8:
                    print("groupIndex>8,跳出循环")
                    break
                if len(newBBlock) > 1:  # 只处理长度大于1的基本块列表
                    bBlockHashList = []  # 基本块哈希列表
                    hashBlockDict = {}  # 创建哈希值和基本块的对应关系字典
                    for eachBlock in newBBlock:
                        bBlockHashList.append(hash(eachBlock))
                        hashBlockDict[hash(eachBlock)] = eachBlock
                    #            print("哈希字典为：",hashBlockDict)
                    #            print("分隔后的基本块的哈希值列表为：",bBlockHashList)
                    bBlockHashListTuple = tuple(bBlockHashList)  # 原基本块哈希值列表
                    #            print("原哈希基本块的顺序为：",bBlockHashListTuple)
                    bBlockPerm = list(itertools.permutations(bBlockHashList))  # 对基本块的哈希值列表排列
                    #            bBlockPerm=list(itertools.permutations(newBBlock)) #对基本块列表排列
                    bBlockPerm.sort()  # 字典排序
                    print("cover的基本块顺序特征为：", bBlockPerm.index(bBlockHashListTuple))
                    eachCoverBinFeature.append(bBlockPerm.index(bBlockHashListTuple))  # 载体的基本块顺序特征
                    if insertIndex.count(groupIndex) == 0:
                        print("不嵌入秘密信息")
                        eachStegoBinFeature.append(bBlockPerm.index(bBlockHashListTuple))
                    else:
                        print("嵌入秘密信息")
                        blockNumFactorial = math.factorial(len(newBBlock))  # 基本块数目的阶乘
                        #                    print("基本块的数目为：",len(newBBlock))
                        blockNumLog = math.log(blockNumFactorial) / math.log(2)  # 阶乘的对数
                        #                print(varlog)
                        if blockNumLog - int(blockNumLog) == 0:
                            bitCount = int(blockNumLog)  # 可以存储的位数刚好为整数
                            permIndex = int(enMessage[0:bitCount], base=2)  # 排列索引
                            print("可以存储1位", "位数：", bitCount)
                            print("排列的索引为：", permIndex)
                        else:  # 可以存储的位数不为整数，需要判断存储的位数
                            if int(enMessage[0:int(blockNumLog) + 1], base=2) > (2 ** int(blockNumLog) - 1):
                                bitCount = int(blockNumLog)
                                permIndex = int(enMessage[0:bitCount], base=2) + blockNumFactorial - 2 ** int(
                                    blockNumLog)  # 排列索引
                            #                            print("可以存储的位数不为整数,且不能多存一位,存储的位数为：",bitCount)
                            #                            print("排列索引为：",permIndex)
                            else:
                                bitCount = int(blockNumLog) + 1
                                permIndex = int(enMessage[0:bitCount], base=2)
                        blockHashAfterPerm = list(bBlockPerm[permIndex])  # 排列之后的基本块哈希值列表
                        #            print("排列后的基本块的哈希值顺序为：",blockHashAfterPerm)
                        blockAfterPerm = []  # 排列之后的基本块列表
                        for blockHash in blockHashAfterPerm:
                            blockAfterPerm.append(hashBlockDict[blockHash])
                        #            print("对应的原基本块为:",blockAfterPerm)
                        print("stego的基本块的顺序特征为:", bBlockPerm.index(tuple(blockHashAfterPerm)))
                        eachStegoBinFeature.append(bBlockPerm.index(tuple(blockHashAfterPerm)))
                        print("原enMessage的长度为：", len(enMessage))
                        enMessage = enMessage[bitCount:len(enMessage)]
                        print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverBinFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverBinFeature:
                    strCoverFeature += bin(var)[2:len(bin(var))]
                #            print("strCoverFeature为：",strCoverFeature)
                #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoBinFeature:
                    strStegoFeature += bin(var)[2:len(bin(var))]
                #            print("steStegoFeature为：",strStegoFeature)
                #            print("eachStegoFeature为：",int(strStegoFeature,base=2))
                coverBBlockFeature.append(int(strCoverFeature, base=2))
                stegoBBlockFeature.append(int(strStegoFeature, base=2))
    return coverBBlockFeature, stegoBBlockFeature


def get_block_zeroOne_feature():
    coverBBlockFeature = []  # cover基本块的顺序特征列表
    stegoBBlockFeature = []  # stego基本块的顺序特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 13212):
        print("正在处理第", x, "个binruntime文件")
        if len(enMessage) < 2000:
            print("enMessage长度不够！！！")
            enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
        eachCoverBinFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoBinFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值

        # 读取binRunCode文件
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
            binRunCode = fp.read()
        #        print("第",x,"个binRuntime文件内容为:",binRunCode)
        # if not binRunCode.strip():
        #     print("第", x, "个binRuntime文件为空")
        if binRunCode.strip():  # 如果匹配到的binRuntime文件不为空
            # 去掉尾部swarm哈希代码
            binRunCode = binRunCode[0:binRunCode.find('a165627a7a72305820')]
            # 获取基本块开始索引
            index575b = binRunCode.find('575b')
            #            print("575b的开始位置为：",index575b)
            # 获取所有基本块代码
            allBlockCode = binRunCode[index575b + 2:len(binRunCode)]
            allBlockGhiDic = get_PC_Inst_GhiDict(allBlockCode)
            allBlockGhiCode = ''
            for eachGhiInst in allBlockGhiDic.values():
                allBlockGhiCode = allBlockGhiCode + eachGhiInst
            blockGhiList = re.findall(r'[I\dabcdef][^GHIJKL]*[GHJKL\dabcdef]', allBlockGhiCode)

            # 获取基本块列表
            blockList = []
            for blockGhiIndex, eachGhiBlock in enumerate(blockGhiList):
                #                print("第",blockGhiIndex+1,"个Ghi基本块为：",eachGhiBlock)
                eachBlock = eachGhiBlock
                for eachGhi in splitGhiList:
                    if eachGhiBlock.count(eachGhi) != 0:
                        eachBlock = eachBlock.replace(eachGhi, splitInstDict2[eachGhi])
                blockList.append(eachBlock)
            #            for blockIndex,eachBlock in enumerate(blockList):
            #                print("第",blockIndex+1,"个基本块为：",eachBlock)
            #            print("匹配到的基本块数目为：",len(blockList))

            #            blockCount=len(blockList)
            #            groupCount=math.ceil(blockCount/8)
            #            print("基本块列表可以分为",groupCount,"组")
            blockList = splitList(blockList, 8)
            for groupIndex, newBBlock in enumerate(blockList):  # 对每组基本块进行处理
                #                print("groupIndex为：",groupIndex)
                if groupIndex > 8:
                    # print("groupIndex>8,跳出循环")
                    break
                if len(newBBlock) > 1:  # 只处理长度大于1的基本块列表
                    bBlockHashList = []  # 基本块哈希列表
                    #                    hashBlockDict={} #创建哈希值和基本块的对应关系字典

                    # 获取bitCount和permIndex
                    blockNumFactorial = math.factorial(len(newBBlock))  # 基本块数目的阶乘
                    #                    print("基本块的数目为：",len(newBBlock))
                    blockNumLog = math.log(blockNumFactorial) / math.log(2)  # 阶乘的对数
                    bitCount = int(blockNumLog)
                    #                    print("最低可嵌入的位数为:",bitCount)

                    # 获取eachCoverBinFeature
                    for eachBlock in newBBlock:
                        bBlockHashList.append(hash(eachBlock))
                    #                        hashBlockDict[hash(eachBlock)]=eachBlock
                    #            print("哈希字典为：",hashBlockDict)
                    #                    print("分隔后的基本块的哈希值列表为：",bBlockHashList)
                    bBlockHashListTuple = tuple(bBlockHashList)  # 原基本块哈希值列表
                    #            print("原哈希基本块的顺序为：",bBlockHashListTuple)
                    bBlockPerm = list(itertools.permutations(bBlockHashList))  # 对基本块的哈希值列表排列
                    bBlockPerm.sort()  # 字典排序
                    formatControlLow = '{:0' + str(bitCount) + 'b}'
                    formatControlHigh = '{:0' + str(bitCount + 1) + 'b}'
                    coverPermIndex = bBlockPerm.index(bBlockHashListTuple)
                    # 根据排列索引求出二进制串
                    if coverPermIndex < (blockNumFactorial - 2 ** bitCount) * 2:  # 嵌入bitCount+1位
                        eachCoverFeature = formatControlHigh.format(coverPermIndex)
                        #                        print("可以嵌入bitCount+1位：",bitCount+1)
                        #                        print("eachCoverFeature为：",eachCoverFeature)
                        eachCoverBinFeature.append(eachCoverFeature)
                    else:  # 嵌入bitCount位
                        eachCoverFeature = formatControlLow.format(coverPermIndex - blockNumFactorial + 2 ** bitCount)
                        #                        print("可以嵌入bitCount位：",bitCount)
                        #                        print("eachCoverFeature为：",eachCoverFeature)
                        eachCoverBinFeature.append(eachCoverFeature)

                        # 获取eachStegoBinFeature
                    if int(enMessage[0:bitCount + 1], base=2) < (blockNumFactorial - 2 ** bitCount) * 2:
                        stegoBitCount = bitCount + 1
                    else:
                        stegoBitCount = bitCount
                    #                    print("eachStegoFeature为：",enMessage[0:stegoBitCount])
                    eachStegoBinFeature.append(enMessage[0:stegoBitCount])

                    # 获取enMessage
                    #                    print("原enMessage的长度为：",len(enMessage))
                    enMessage = enMessage[stegoBitCount:len(enMessage)]
            #                    print("新enMessage的长度为：",len(enMessage))
            if len(eachCoverBinFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverBinFeature:
                    strCoverFeature += var
                #                print("第",x,"个binRuntime文件的strCoverFeature为：",strCoverFeature)
                coverZerolen = strCoverFeature.count('0')
                coverOneLen = strCoverFeature.count('1')
                coverFeatureValue = (coverZerolen - coverOneLen) / (float(coverZerolen) + coverOneLen)
                #                print("coverFeatureValue为：",coverFeatureValue)
                coverBBlockFeature.append(coverFeatureValue)
                #            print("strCoverFeature为：",strCoverFeature)
                #            print("eachCoverFeature为：",int(strCoverFeature,base=2))
                for var in eachStegoBinFeature:
                    strStegoFeature += var
                #                print("第",x,"个binRuntime文件的strStegoFeature为：",strStegoFeature)
                stegoZerolen = strStegoFeature.count('0')
                stegoOneLen = strStegoFeature.count('1')
                stegoFeatureValue = (stegoZerolen - stegoOneLen) / (float(stegoZerolen) + stegoOneLen)
                #                print("stegoFeatureValue为:",stegoFeatureValue)
                stegoBBlockFeature.append(stegoFeatureValue)
    return coverBBlockFeature, stegoBBlockFeature


def get_block_zeroOneRandom_feature():
    coverBBlockFeature = []  # cover基本块的顺序特征列表
    stegoBBlockFeature = []  # stego基本块的顺序特征列表
    with open(base_message_path, 'r', encoding='utf-8') as fp:
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    for x in range(1, 13212):
        print("开始处理第", x, "个binRuntime")
        eachCoverBinFeature = []  # 每个Cover合约中的所有组的特征值列表，最终需要将此列表转为一个特征值
        eachStegoBinFeature = []  # 每个Stego合约中的所有组的特征值列表，最终需要将此列表转为一个特征值

        # 读取binRunCode文件
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
            binRunCode = fp.read()
        #        print("第",x,"个binRuntime文件内容为:",binRunCode)
        # if not binRunCode.strip():
        #     print("第", x, "个binRuntime文件为空")
        if binRunCode.strip():  # 如果匹配到的binRuntime文件不为空
            # 去掉尾部swarm哈希代码
            binRunCode = binRunCode[0:binRunCode.find('a165627a7a72305820')]
            # 获取基本块开始索引
            index575b = binRunCode.find('575b')
            #            print("575b的开始位置为：",index575b)
            # 获取所有基本块代码
            allBlockCode = binRunCode[index575b + 2:len(binRunCode)]
            allBlockGhiDic = get_PC_Inst_GhiDict(allBlockCode)
            allBlockGhiCode = ''
            for eachGhiInst in allBlockGhiDic.values():
                allBlockGhiCode = allBlockGhiCode + eachGhiInst
            blockGhiList = re.findall(r'[I\dabcdef][^GHIJKL]*[GHJKL\dabcdef]', allBlockGhiCode)

            # 获取基本块列表
            blockList = []
            for blockGhiIndex, eachGhiBlock in enumerate(blockGhiList):
                #                print("第",blockGhiIndex+1,"个Ghi基本块为：",eachGhiBlock)
                eachBlock = eachGhiBlock
                for eachGhi in splitGhiList:
                    if eachGhiBlock.count(eachGhi) != 0:
                        eachBlock = eachBlock.replace(eachGhi, splitInstDict2[eachGhi])
                blockList.append(eachBlock)
            #            for blockIndex,eachBlock in enumerate(blockList):
            #                print("第",blockIndex+1,"个基本块为：",eachBlock)
            #            print("匹配到的基本块数目为：",len(blockList))

            blockCount = len(blockList)
            groupCount = math.ceil(blockCount / 8)
            # print("基本块列表可以分为", groupCount, "组")

            # 使用随机数种子决定是否嵌入数据
            random.seed(9)
            dataList = list(range(groupCount))
            insertIndex = []  # 选出嵌入索引
            for i in range(int(groupCount / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            #            print("insertIndex为：",insertIndex)
            blockList = splitList(blockList, 8)  # 将基本块分为8个一组
            for groupIndex, newBBlock in enumerate(blockList):  # 对每组基本块进行处理
                #            print("分隔后的基本块列表为：",newBBlock)
                # print("groupIndex为：", groupIndex)
                if groupIndex > 8:
                    # print("groupIndex>8,跳出循环")
                    break
                if len(newBBlock) > 1:  # 只处理长度大于1的基本块列表
                    bBlockHashList = []  # 基本块哈希列表
                    hashBlockDict = {}  # 创建哈希值和基本块的对应关系字典

                    # 获取bitCount和permIndex
                    blockNumFactorial = math.factorial(len(newBBlock))  # 基本块数目的阶乘
                    #                    print("基本块的数目为：",len(newBBlock))
                    blockNumLog = math.log(blockNumFactorial) / math.log(2)  # 阶乘的对数
                    # bitCount表示最低嵌入的位数
                    bitCount = int(blockNumLog)

                    # 获取eachCoverBinFeature
                    for eachBlock in newBBlock:
                        bBlockHashList.append(hash(eachBlock))
                        hashBlockDict[hash(eachBlock)] = eachBlock
                    #            print("哈希字典为：",hashBlockDict)
                    #            print("分隔后的基本块的哈希值列表为：",bBlockHashList)
                    bBlockHashListTuple = tuple(bBlockHashList)  # 原基本块哈希值列表
                    #            print("原哈希基本块的顺序为：",bBlockHashListTuple)
                    bBlockPerm = list(itertools.permutations(bBlockHashList))  # 对基本块的哈希值列表排列
                    bBlockPerm.sort()  # 字典排序
                    formatControlLow = '{:0' + str(bitCount) + 'b}'
                    formatControlHigh = '{:0' + str(bitCount + 1) + 'b}'
                    coverPermIndex = bBlockPerm.index(bBlockHashListTuple)
                    # 根据排列索引求出二进制串
                    if coverPermIndex < (blockNumFactorial - 2 ** bitCount) * 2:  # 嵌入bitCount+1位
                        eachCoverFeature = formatControlHigh.format(coverPermIndex)
                        # print("eachCoverFeature为：", eachCoverFeature)
                        eachCoverBinFeature.append(eachCoverFeature)
                    else:  # 嵌入bitCount位
                        eachCoverFeature = formatControlLow.format(coverPermIndex - blockNumFactorial + 2 ** bitCount)
                        # print("eachCoverFeature为：", eachCoverFeature)
                        eachCoverBinFeature.append(eachCoverFeature)

                        # 获取eachStegoBinFeature
                    if insertIndex.count(groupIndex) == 0:
                        # print("不嵌入秘密信息")
                        # print("eachStegoBinFeature的值为：", eachCoverFeature)
                        eachStegoBinFeature.append(eachCoverFeature)
                    else:
                        # print("嵌入秘密信息")
                        if int(enMessage[0:bitCount + 1], base=2) < (blockNumFactorial - 2 ** bitCount) * 2:
                            stegoBitCount = bitCount + 1
                        else:
                            stegoBitCount = bitCount
                        # print("eachStegoFeature为：", enMessage[0:stegoBitCount])
                        eachStegoBinFeature.append(enMessage[0:stegoBitCount])
                        # 处理enMessage
                        # print("原enMessage的长度为：", len(enMessage))
                        enMessage = enMessage[stegoBitCount:len(enMessage)]
                        # print("新enMessage的长度为：", len(enMessage))
            if len(eachCoverBinFeature) > 0:
                strCoverFeature = ''
                strStegoFeature = ''
                for var in eachCoverBinFeature:
                    strCoverFeature += var
                #            print("strCoverFeature为：",strCoverFeature)
                coverZerolen = strCoverFeature.count('0')
                coverOneLen = strCoverFeature.count('1')
                coverBBlockFeature.append((coverZerolen - coverOneLen) / (float(coverZerolen) + coverOneLen))
                for var in eachStegoBinFeature:
                    strStegoFeature += var
                #            print("steStegoFeature为：",strStegoFeature)
                stegoZerolen = strStegoFeature.count('0')
                stegoOneLen = strStegoFeature.count('1')
                stegoBBlockFeature.append((stegoZerolen - stegoOneLen) / (float(stegoZerolen) + stegoOneLen))
    return coverBBlockFeature, stegoBBlockFeature


def train():
    # coverBBlockFeature,stegoBBlockFeature=get_block_feature()
    # coverBBlockFeature,stegoBBlockFeature=get_blockRandom_feature()
    # coverBBlockFeature, stegoBBlockFeature = get_block_zeroOne_feature()
    coverBBlockFeature, stegoBBlockFeature = get_block_zeroOneRandom_feature()
    X = coverBBlockFeature + stegoBBlockFeature  # 获取所有的特征
    X = np.array(X).reshape(-1, 1)
    print("所有的特征长度为：", len(X))
    lenCover = int(len(coverBBlockFeature))
    lenStego = int(len(stegoBBlockFeature))
    print("Cover特征数组的长度为：", lenCover)
    print("Stego特征数组的长度为", lenStego)
    y = [0] * lenCover + [1] * lenStego  # 获取所有的标签
    print("所有的标签长度为：", len(y))
    train_X, test_X, train_y, test_true_y = train_test_split(X, y, test_size=0.3, random_state=0)
    svm_model_rbf = svm.SVC(kernel='rbf', gamma=0.235, C=109.)  # 高斯核
    svm_model_rbf.fit(train_X, train_y)
    test_predict_rbf_y = svm_model_rbf.predict(test_X)  # 预测的真实值
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))
    con_matrix = confusion_matrix(y_true=test_true_y, y_pred=test_predict_rbf_y)
    print("混淆矩阵为：", con_matrix)
    print("FP rate:", 100 * con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1]), "%")
    print("FN rate:", 100 * con_matrix[0][1] / (con_matrix[0][0] + con_matrix[0][1]), "%")
    drawHist(coverBBlockFeature, stegoBBlockFeature)


if __name__ == '__main__':
    train()
#    coverBBlockFeature,stegoBBlockFeature=getFeature()
#    print("cover特征的长度为：",len(coverBBlockFeature))
#    print("cover特征为：",coverBBlockFeature)
#    print("stego特征的长度为：",len(stegoBBlockFeature))
#    print("stego特征为：",stegoBBlockFeature)
#    get_blockRandom_feature()
