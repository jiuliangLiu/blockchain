# -*- coding: utf-8 -*-
import re
import numpy as np
# import pandas as pd
from sklearn import svm
from function_test import aes_en, encodeToBinStr  # 导入加密等函数
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# array=np.random.randn(5,4)
# print(array)
# df_obj=pd.DataFrame(array)
# print(df_obj.head())
base_read_path = 'G:\Python_Crawler\coverContracts(no_annotation)\coverContract{}.sol'
base_message_path = 'G:\实验\秘密信息\message1.txt'  # 秘密信息存储路径


def word_count_in_str(string, keyword):
    return len(string.split(keyword)) - 1


def geEnMessage():
    with open(base_message_path, 'r', encoding='utf-8') as fp:  # 读取秘密信息
        message = fp.read()
    enMessage = encodeToBinStr(aes_en(message)).replace(' ', '')  # 加密后的秘密信息的二进制表示
    return enMessage


def getFeature():  # 分别获取cover和stego的uint特征
    coverUintFeature = []  # 载体合约中的uint和uint256特征列表
    stegoUintFeature = []  # 含秘载体合约中的uint和uint256特征列表
    enMessage = geEnMessage()
    for x in range(1, 5000):
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


def getTotalUintFeature():  # 获取所有的特征
    coverUintFeature, stegoUintFeature = getFeature()
    totalUintFeature = coverUintFeature + stegoUintFeature
    return totalUintFeature


def getLabels():  # 返回所有特征对应的标签
    coverUintFeature, stegoUintFeature = getFeature()
    coverLabels = [0] * len(coverUintFeature)
    stegoLabels = [1] * len(stegoUintFeature)
    totalLabels = coverLabels + stegoLabels
    return totalLabels


def train():
    X = getTotalUintFeature()  # 训练特征
    #    print("总特征集为：",X)
    y = getLabels()  # 训练标签
    #    print("对应的总标签为：",y)
    train_X, test_X, train_y, test_true_y = train_test_split(X, y, test_size=0.3, random_state=0)
    #    print("训练集的特征为：",train_X)
    #    print("测试集的特征为：",test_X)
    #    print("训练集的标签为：",train_y)
    #    print("测试集的标签真实值为：",test_true_y)
    #    print(type(test_true_y))
    train_X = np.array(train_X).reshape(-1, 1)
    test_X = np.array(test_X).reshape(-1, 1)
    #    svm_model_linear=svm.SVC(kernel='linear',C=100.) #线性核
    #    svm_model_linear.fit(train_X,train_y)
    #    test_predict_linear_y=svm_model_linear.predict(test_X) #预测的真实值
    #    print("SVM线性核预测的准确度为：",accuracy_score(test_true_y,test_predict_linear_y))
    model = svm.SVC(kernel='rbf')
    c_can = np.linspace(100, 110, 10)
    gamma_can = np.linspace(0.1, 0.5, 10)
    svc_rbf = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
    svc_rbf.fit(train_X, train_y)
    print('最优参数：\n', svc_rbf.best_params_)
    svm_model_rbf = svm.SVC(kernel='rbf', gamma=0.25, C=100.)  # 线性核
    svm_model_rbf.fit(train_X, train_y)
    test_predict_rbf_y = svm_model_rbf.predict(test_X)  # 预测的真实值
    print("SVM高斯核预测的准确度为：", accuracy_score(test_true_y, test_predict_rbf_y))


#    svm_model_poly=svm.SVC(kernel='poly',degree=3,C=100.) #线性核
#    svm_model_poly.fit(train_X,train_y) 
#    test_predict_poly_y=svm_model_poly.predict(test_X) #预测的真实值
#    print("SVM多项式核预测的准确度为：",accuracy_score(test_true_y,test_predict_poly_y))

if __name__ == '__main__':
    train()
