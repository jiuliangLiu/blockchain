# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:01:30 2019

@author: 刘九良
"""

import re
import numpy as np
# import pandas as pd
from sklearn import svm
from function_test import aes_en, encodeToBinStr  # 导入加密等函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from svm_uint_test import word_count_in_str, geEnMessage, drawHist
from svm_bBlock_test import splitList
from sklearn.model_selection import GridSearchCV
import random
import pandas as pd

base_read_path = r'F:\experimentData\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path = r'F:\experimentData\message1.txt'  # 秘密信息存储路径


def get_equalSubAll_feature():
    coverUintFeature = []
    stegoUintFeature = []
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    # 以下循环处理20000个合约
    for x in range(1, 20000):
        pattern1Dic = {}
        pattern2Dic = {}
        pattern3Dic = {}
        equalSubDic = {}
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)

        # 匹配合约中的所有等价语句
        pattern1 = re.compile(r'\b(u)?int(256)?\b')
        matchUint = pattern1.finditer(source_code)
        coverUintLen = 0
        coverUint256Len = 0
        for eachUint in matchUint:
            pattern1Dic[eachUint.start()] = eachUint.group()
            if eachUint.group() == 'uint':
                coverUintLen += 1
            else:
                coverUint256Len += 1
        pattern1Values = list(pattern1Dic.values())

        # 获取coverUint特征
        if coverUintLen + coverUint256Len > 0:
            coverUintFeature.append((coverUint256Len - coverUintLen) / (float(coverUintLen) + coverUint256Len))

        pattern2 = re.compile(r'\b([\w.\[\]]+?)\s*(>=|<=|>|<|==|!=|\+=|-=|\*=|/=|%='
                              r'|&=|\|=|\^=|&&|\|\||&|\|)\s*([\w.\[\]]+?)\b')
        matchPattern2 = pattern2.finditer(source_code)
        for eachEqualState in matchPattern2:
            pattern2Dic[eachEqualState.start()] = eachEqualState.group()

        pattern3 = re.compile(r'\b([\w]+\+\+)|([\w]+--)|(\+\+[\w]+)|(--[\w]+)|(\".*\")|(\'.*\')\b')
        matchPattern3 = pattern3.finditer(source_code)
        for eachPattern3 in matchPattern3:
            pattern3Dic[eachPattern3.start()] = eachPattern3.group()

        equalSubDic.update(pattern1Dic)
        equalSubDic.update(pattern2Dic)
        equalSubDic.update(pattern3Dic)
        equalSubSeries = pd.Series(equalSubDic)
        equalSubDic = dict(equalSubSeries)
        #        print("字典的长度为：",len(equalSubDic))
        subEnMessage = enMessage[0:len(equalSubDic)]
        print("原enMessage的长度为：", len(enMessage))
        enMessage = enMessage[len(equalSubDic):len(enMessage)]
        print("新enMessage的长度为：", len(enMessage))
        subEnMessageIndex = 0
        stegoUintLen = 0
        stegoUint256Len = 0
        for key in equalSubDic:
            if pattern1Values.count(equalSubDic[key]) != 0:
                if subEnMessage[subEnMessageIndex] == '0':
                    stegoUintLen += 1
                else:
                    stegoUint256Len += 1
            subEnMessageIndex += 1
        if stegoUintLen + stegoUint256Len > 0:
            stegoUintFeature.append((stegoUint256Len - stegoUintLen) / (float(stegoUint256Len) + stegoUintLen))
    return coverUintFeature, stegoUintFeature


def get_equalSub_feature():
    coverEqualSubFeature = []
    stegoEqualSubFeature = []
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    # 以下循环处理20000个合约
    for x in range(1, 20000):
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        # 匹配合约中的所有等价语句
        pattern = re.compile(r'\b([\w.\[\]]+?)\s*(>=|<=|>|<|==|!=|\+=|-=|\*=|/=|%='
                             r'|&=|\|=|\^=|&&|\|\||&|\|)\s*([\w.\[\]]+?)\b')
        #        pattern = re.compile(r'\b([\w.\[\]]+?)\s*(>=)\s*([\w.\[\]]+?)\b')
        matchEqualState = pattern.findall(source_code)
        #        print("matchEqualState:",matchEqualState)
        #        matchFunction=re.findall(r'function\s[^\(\)]+\([^\(\)]+,[^\(\)]+\)',source_code)
        #        print("第",x,"个合约匹配到的等价语句为：",matchEqualState)
        #        print("第",x,"个合约匹配到的函数为：",matchFunction)
        #        print("第",x,"个合约匹配到的等价语句的个数为：",len(matchEqualState))
        zeroLen = 0
        oneLen = 0
        for var in matchEqualState:
            #            print(len(var))
            #            print("var[0]为：",var[0])
            #            print("var[2]为：",var[2])
            if hash(var[0]) < hash(var[2]):
                zeroLen = zeroLen + 1
            else:
                oneLen = oneLen + 1
        totalLen = zeroLen + oneLen
        if totalLen > 0:
            coverEqualSubFeature.append((zeroLen - oneLen) / (float(zeroLen) + oneLen))
            #        print("coverEqualSubFeature为：",coverEqualSubFeature)
            subEnMessage = enMessage[0:int(totalLen)]
            #            print("enMessage的原长度为：",len(enMessage))
            stegoZeroLen = word_count_in_str(subEnMessage, '0')
            stegoOneLen = word_count_in_str(subEnMessage, '1')
            stegoEqualSubFeature.append(
                (stegoZeroLen - stegoOneLen) / (float(stegoZeroLen) + stegoOneLen))  # stego的uint特征
            enMessage = enMessage[int(totalLen):len(enMessage)]  # 获取剩余的消息
            print("剩余的enMessage长度为", len(enMessage))
    return coverEqualSubFeature, stegoEqualSubFeature


def get_equalSubRandom_feature():
    coverEqualSubFeature = []
    stegoEqualSubFeature = []
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 获取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')
    # 以下循环处理20000个合约
    for x in range(1, 20000):
        read_path = base_read_path.format(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        # 匹配合约中的所有等价语句
        pattern = re.compile(r'\b([\w.\[\]]+?)\s*(>=|<=|>|<|==|!=|\+=|-=|\*=|/=|%='
                             r'|&=|\|=|\^=|&&|\|\||&|\|)\s*([\w.\[\]]+?)\b')
        matchEqualState = pattern.findall(source_code)
        zeroLen = 0
        oneLen = 0
        totalLen = len(matchEqualState)
        #        print("totalLen为：",totalLen)
        # 用于计算cover特征
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
            coverEqualSubFeature.append((zeroLen - oneLen) / (float(zeroLen) + oneLen))
            #        print("coverEqualSubFeature为：",coverEqualSubFeature)

            # 使用随机数种子选取嵌入点
            random.seed(9)
            dataList = list(range(int(totalLen)))
            insertIndex = []

            # 选出嵌入索引
            for i in range(int(totalLen / 2)):
                randIndex = int(random.uniform(0, len(dataList)))
                insertIndex.append(dataList[randIndex])
                del (dataList[randIndex])
            # print("insertIndex为", insertIndex)
            # print("insertIndex的长度为：", len(insertIndex))
            subEnMessage = enMessage[0:int(len(insertIndex))]
            # 根据随机数种子计算stego特征
            stegoZeroLen = 0
            stegoOneLen = 0
            subMessageIndex = 0
            for stateIndex, var in enumerate(matchEqualState):
                #                print("第",stateIndex,"条语句")
                if insertIndex.count(stateIndex) == 0:  # 不嵌入信息
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
            # print("enMessage的原长度为：", len(enMessage))
            stegoEqualSubFeature.append(
                (stegoZeroLen - stegoOneLen) / (float(stegoZeroLen) + stegoOneLen))  # stego的uint特征
            enMessage = enMessage[int(len(insertIndex)):len(enMessage)]  # 获取剩余的消息
            print("剩余的enMessage长度为", len(enMessage))
    return coverEqualSubFeature, stegoEqualSubFeature


def train():
    # coverEqualSubFeature, stegoEqualSubFeature = get_equalSub_feature()
    coverEqualSubFeature, stegoEqualSubFeature = get_equalSubRandom_feature()
    # coverEqualSubFeature, stegoEqualSubFeature = get_equalSubAll_feature()
    X = coverEqualSubFeature + stegoEqualSubFeature  # 训练特征
    X = np.array(X).reshape(-1, 1)
    #    print("总特征集为：",X)
    #    print("总特征集的长度为：",len(X))
    y = [0] * len(coverEqualSubFeature) + [1] * len(stegoEqualSubFeature)  # 训练标签
    #    print("对应的总标签为：",y)
    #    print("对应的总标签的长度为：",len(y))
    train_X, test_X, train_y, test_true_y = train_test_split(X, y, test_size=0.3, random_state=0)

    # rbf核预测，网格搜索最优参数
    #    model=svm.SVC(kernel='rbf')
    #    c_can=np.linspace(100,104,10)
    #    gamma_can = np.linspace(0.3, 0.4, 10)
    #    svc_rbf = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
    #    svc_rbf.fit(train_X,train_y)
    #    print ('最优参数：\n', svc_rbf.best_params_)
    #    test_predict_rbf_y=svc_rbf.predict(test_X) #预测值
    #    print("预测值为：",test_predict_rbf_y.reshape(-1,1))
    #    print("SVM高斯核预测的准确度为：",accuracy_score(test_true_y,test_predict_rbf_y))

    #    rbf核预测
    svm_model_rbf = svm.SVC(kernel='rbf', gamma=0.235, C=109.)  # 高斯核
    svm_model_rbf.fit(train_X, train_y)
    test_predict_rbf_y = svm_model_rbf.predict(test_X)  # 预测值
    #    print("预测值为：",test_predict_rbf_y.reshape(-1,1))
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))
    con_matrix = confusion_matrix(y_true=test_true_y, y_pred=test_predict_rbf_y)
    print("混淆矩阵为：", con_matrix)
    print("FP rate:", 100 * con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1]), "%")
    print("FN rate:", 100 * con_matrix[0][1] / (con_matrix[0][0] + con_matrix[0][1]), "%")
    #    print("coverEqualSubFeature",coverEqualSubFeature)
    drawHist(coverEqualSubFeature, stegoEqualSubFeature)


if __name__ == '__main__':
    train()
#    coverEqualSubFeature,stegoEqualSubFeature=get_equalSub_feature()
#    print("coverEqualSubFeature",coverEqualSubFeature)
#    print("coverEqualSubFeature的长度为：",len(coverEqualSubFeature))
#    print("stegoEqualSubFeature",stegoEqualSubFeature)
#    print("stegoEqualSubFeature的长度为：",len(stegoEqualSubFeature))
#    get_equalSubRandom_feature()
#    get_equalSubAll_feature()
