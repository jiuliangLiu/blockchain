# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:48:05 2019
用SVM检测uint或uint256隐藏信息
@author: 刘九良
"""

# -*- coding: utf-8 -*-
import re
import numpy as np
# import pandas as pd
from sklearn import svm
from function_test import aes_en, encodeToBinStr  # 导入加密等函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import random
import matplotlib.mlab as mlab

base_read_path = r'F:\experimentData\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path = r'F:\experimentData\message1.txt'  # 秘密信息存储路径


def word_count_in_str(string, keyword):
    return len(string.split(keyword)) - 1


def geEnMessage():
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 读取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')  # 加密后的秘密信息的二进制表示
    return enMessage


def get_str_feature():
    coverUintFeature = []  # 载体合约中的uint和uint256特征列表
    stegoUintFeature = []  # 含秘载体合约中的uint和uint256特征列表
    enMessage = geEnMessage()
    for x in range(1, 500):
        read_path = base_read_path.format(x)
        #    print(read_path)
        #    print(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        #        matchUint=re.findall(r'\Wuint\W',source_code)
        #        matchUint256=re.findall(r'\Wuint256\W',source_code)
        pattern = re.compile(r'\buint(256)?\b')
        matchAll = pattern.finditer(source_code)
        uintList = []
        strCoverFeature = ''
        strStegoFeature = ''
        for i, match in enumerate(matchAll):
            #            print("第",i,"个匹配到的值为：",match.group())
            uintList.append(match.group())
            if match.group() == 'uint':
                strCoverFeature = strCoverFeature + '0'
            else:
                strCoverFeature = strCoverFeature + '1'
        if (len(uintList)) > 1:  # 如果匹配到的总数大于1
            totalLen = len(uintList)
            subEnMessage = enMessage[0:totalLen]
            #            print("subEnMessage为：",subEnMessage)
            strStegoFeature = subEnMessage
            coverUintFeature.append(int(strCoverFeature, base=2))
            stegoUintFeature.append(int(strStegoFeature, base=2))
            print("enMessage的原长度为：", len(enMessage))
            enMessage = enMessage[totalLen:len(enMessage)]  # 获取剩余的消息
            print("剩余的enMessage长度为", len(enMessage))
    #        print("coverUintFeature",coverUintFeature)
    #        print("stegoUintFeature",stegoUintFeature)
    return coverUintFeature, stegoUintFeature


def get_uint_zeroOne_feature():
    coverUintFeature = []  # 载体合约中的uint和uint256特征列表
    stegoUintFeature = []  # 含秘载体合约中的uint和uint256特征列表
    enMessage = geEnMessage()
    for x in range(1, 20000):
        read_path = base_read_path.format(x)
        #    print(read_path)
        #    print(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        matchUint = re.findall(r'\Wuint\W', source_code)
        matchUint256 = re.findall(r'\Wuint256\W', source_code)
        #        print('第',x,'个合约：')
        #    print('uint的总数为：',len(matchUint))
        #    print(matchUint)
        #    print('uint256的总数为：',len(matchUint256))
        #    print(matchUint256)
        coverUintLen = len(matchUint)
        coverUint256Len = len(matchUint256)
        totalLen = float(coverUintLen) + coverUint256Len
        if (totalLen) > 0:  # 如果包含uint或uint256，则可以提出特征
            coverUintFeature.append((coverUint256Len - coverUintLen) / (float(coverUintLen) + coverUint256Len))
            subEnMessage = enMessage[0:int(totalLen)]
            #            print("要去除的长度为：",int(totalLen),len(subEnMessage))
            #            print("enMessage的原长度为：",len(enMessage))
            stegoUintLen = word_count_in_str(subEnMessage, '0')
            stegoUint256Len = word_count_in_str(subEnMessage, '1')
            stegoUintFeature.append(
                (stegoUint256Len - stegoUintLen) / (float(stegoUintLen) + stegoUint256Len))  # stego的uint特征
            enMessage = enMessage[int(totalLen):len(enMessage)]  # 获取剩余的消息
            print("剩余的enMessage长度为", len(enMessage))
    #        print("子串为：",subEnMessage)
    #        print("子串中的0个数为：",word_count_in_str(subEnMessage,'0'))
    #        print("子串中的1个数为：",word_count_in_str(subEnMessage,'1'))
    #    print("coverUintFeature的长度为：",len(coverUintFeature))
    #    print("stegoUintFeature的长度为：",len(stegoUintFeature))
    #    totalUintFeature=coverUintFeature+stegoUintFeature
    #    print("totalUintFeature的长度为：",len(totalUintFeature))
    return coverUintFeature, stegoUintFeature


def get_uintRandom_zeroOne_feature():
    coverUintFeature = []  # 载体合约中的uint和uint256特征列表
    stegoUintFeature = []  # 含秘载体合约中的uint和uint256特征列表
    enMessage = geEnMessage()
    for x in range(1, 20000):
        read_path = base_read_path.format(x)
        #    print(read_path)
        #    print(x)
        with open(read_path, 'r', encoding='utf-8') as fp:
            source_code = fp.read()
            # print(source_code)
        #        matchUint=re.findall(r'\Wuint\W',source_code)
        #        matchUint256=re.findall(r'\Wuint256\W',source_code)
        pattern = re.compile(r'\buint(256)?\b')
        matchAll = pattern.finditer(source_code)
        uintList = []
        for i, match in enumerate(matchAll):
            #            print("第",i,"个匹配到的值为：",match.group())
            uintList.append(match.group())
        if (len(uintList)) > 1:  # 如果匹配到的总数大于1
            #            print("原uintList为：",uintList)
            coverUintLen = uintList.count('uint')
            coverUint256Len = uintList.count('uint256')
            totalLen = len(uintList)
            #            print("uint个数为：",uintList.count('uint'))
            #            print("uint256的个数为：",uintList.count('uint256'))
            if totalLen > 0:  # cover载体合约中的总数大于0
                coverUintFeature.append((coverUint256Len - coverUintLen) / (float(coverUintLen) + coverUint256Len))
                # 使用随机数种子选取嵌入点
                random.seed(9)
                dataList = list(range(int(totalLen)))
                insertIndex = []  # 选出嵌入索引

                # 以下循环用于获取嵌入索引
                for i in range(int(totalLen / 2)):
                    randIndex = int(random.uniform(0, len(dataList)))
                    insertIndex.append(dataList[randIndex])
                    del (dataList[randIndex])
                #            print("总长度为：",totalLen)
                #                print("insertIndex为",insertIndex)
                #            print("insertIndex的长度为",len(insertIndex))
                #            print("dataList为",dataList)

                subEnMessage = enMessage[0:len(insertIndex)]
                #            print("subEnMessage为：",subEnMessage)

                # 根据嵌入索引修改uintList
                for i, eachIndex in enumerate(insertIndex):
                    #                    print("insertIndex为：",eachIndex)
                    if subEnMessage[i] == '0':
                        #                    print("秘密消息为0")
                        uintList[eachIndex] = 'uint'
                    else:
                        #                    print("秘密消息为1")
                        uintList[eachIndex] = 'uint256'
                #            print("新的uintList为：",uintList)
                stegoUintLen = uintList.count('uint')
                stegoUint256Len = uintList.count('uint256')
                #            print("新uint个数为：",uintList.count('uint'))
                #            print("新uint256的个数为：",uintList.count('uint256'))
                print("enMessage的原长度为：", len(enMessage))
                stegoUintFeature.append(
                    (stegoUint256Len - stegoUintLen) / (float(stegoUintLen) + stegoUint256Len))  # stego的uint特征
                enMessage = enMessage[len(insertIndex):len(enMessage)]  # 获取剩余的消息
                print("剩余的enMessage长度为", len(enMessage))
    return coverUintFeature, stegoUintFeature


def train():
    #    coverUintFeature,stegoUintFeature=get_str_feature()
    # coverUintFeature, stegoUintFeature = get_uint_zeroOne_feature()
    coverUintFeature, stegoUintFeature = get_uintRandom_zeroOne_feature()
    X = coverUintFeature + stegoUintFeature  # 训练特征
    X = np.array(X).reshape(-1, 1)
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
    print("预测值为：", test_predict_rbf_y)
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))
    con_matrix = confusion_matrix(y_true=test_true_y, y_pred=test_predict_rbf_y)
    print("混淆矩阵为：", con_matrix)
    print("FP rate:", 100 * con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1]), "%")
    print("FN rate:", 100 * con_matrix[0][1] / (con_matrix[0][0] + con_matrix[0][1]), "%")
    drawHist(coverUintFeature, stegoUintFeature)


# 画图
#    n_support_vector=svm_model_rbf.n_support_ #支持向量的个数
#    print("支持向量的个数为：",n_support_vector)
#    Support_vector_index = svm_model_rbf.support_ #支持向量索引
#    plot_point(train_X,train_y,Support_vector_index)

# 多项式核预测
#    svm_model_poly=svm.SVC(kernel='poly',degree=3,C=100.) #线性核
#    svm_model_poly.fit(train_X,train_y) 
#    test_predict_poly_y=svm_model_poly.predict(test_X) #预测的真实值
#    print("SVM多项式核预测的准确度为：",accuracy_score(test_true_y,test_predict_poly_y))

def plot_point(dataArr, labelArr, Support_vector_index):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='y', s=20)
    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=100, c='', alpha=0.5, linewidth=1.5, edgecolor='red')
    plt.show()


def drawHist(coverFeature, stegoFeature):
    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(coverFeature, bins=200, density=True, alpha=1, histtype='stepfilled',
             color='blue', edgecolor='none')
    ax2.hist(stegoFeature, bins=200, density=True, alpha=1, histtype='stepfilled',
             color='blue', edgecolor='none')
    #    plt.ylim(0,1)
    font1 = {'family': 'Microsoft YaHei', 'size': 18}
    ax1.set_title('原始合约', fontdict=font1)
    ax1.set_xlabel('特征值', fontdict=font1)
    ax1.set_ylabel('频率', fontdict=font1)
    ax2.set_title('含秘合约', fontdict=font1)
    ax2.set_xlabel('特征值', fontdict=font1)
    ax2.set_ylabel('频率', fontdict=font1)
    # plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


if __name__ == '__main__':
    #    get_random_Feature()
    train()
