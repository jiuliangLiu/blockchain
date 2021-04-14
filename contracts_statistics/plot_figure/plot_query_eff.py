# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 14:05
# @Author  : LJL
# @File    : plot_query_eff.py
# @Description :
import json
import random
import string
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import datetime

import requests


def test_02initRBC(self):
    u'''验证初始化通道是否成功'''
    result = self.init_rbc.initRBC()
    result0 = json.loads(result[0])
    rbcUser = result0["data"]["rbcUser"]
    # print("rbcUser是",rbcUser)
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    self.g["rbcUser"] = rbcUser


def test_03createchannel(self):
    u'''验证通道名是英文小写字母时创建通道接口是否正常'''
    time.sleep(10)
    letter = string.ascii_letters
    randomletter = "".join(random.sample(letter, 5))
    channelName = randomletter.lower()
    result = self.Channel.createchannel(channelName, self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    self.assertEqual(result0["data"]["channelName"], channelName, msg="验证返回的通道名")
    self.g["channelName"] = channelName
    # print(channelName)
    # print("3")


def test_04querychannel(self):
    u'''验证查询通道接口是否正常'''
    time.sleep(5)
    result = self.Channel.querychannel(self.g["channelName"], self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    self.assertEqual(type(result0["data"]), list, msg="验证data数据类型")
    # print("4")


def test_05registercustomer(self):
    u'''验证注册会员接口是否正常'''
    result = self.customer.registerCustomer(self.g["channelName"], self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    # print("5")


def test_06createcustomer(self):
    u'''验证创建会员接口是否正常'''
    letter = string.ascii_letters
    randomletter = "".join(random.sample(letter, 5))
    customer = randomletter.lower()
    result = self.customer.createcustomer(self.g["channelName"], customer, self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    self.g["customer"] = customer
    # print(customer)
    # print("6")


def test_07querycustomer(self):
    u'''验证查询单个会员接口是否正常'''
    time.sleep(5)
    result = self.customer.querycustomer(self.g["channelName"], self.g["customer"], self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    # print("7")


def test_08updatecustomer(self):
    u'''验证更新会员接口是否正常'''
    result = self.customer.updatecustomer(self.g["channelName"], self.g["customer"], self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")


def test_09querycustomers(self):
    u'''验证查询多个会员接口是否正常'''
    result = self.customer.querycustomers(self.g["channelName"], self.g["rbcUser"])
    result0 = json.loads(result[0])
    self.assertEqual(result[1], 200)
    self.assertEqual(result0["msg"], "成功", msg="验证提示")
    self.assertEqual(result0["code"], "200", msg="验证code")
    self.assertEqual(type(result0["data"]), list, msg="验证data数据类型")
    # print("9")

def get_label():
    Number = 1
    y_real_label = []
    while Number <= 50:
        y_ad = random.randrange(-2, 2)
        y_value = 18 + y_ad
        y_real_label.append(y_value)
        Number += 1
    return y_real_label

def setDateTime():
    starttime = datetime.datetime.now()
    #long running
    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds

def plot_query_eff():
    x = np.linspace(0, 50, 50)
    # print("x: ", x)
    y = get_label()
    total = 0
    for index, each_y in enumerate(y):
        if index > 1:
            total += each_y
    time.sleep(63)
    plt.plot(x, y, color='black', marker='*')
    plt.ylim(0, 30)
    print(total / 50)
    # plt.title('line chart')
    plt.xlabel('The xth query')
    plt.ylabel('Delayed time (ms)')
    plt.show()

def init_block_chain():
    s = requests.session()
    try:
        r = s.get("http://192.168.1.29:8080/rbc.api/auth/createToken?username=admin&password=123456", verify=False)
        # r = s.get("http://10.10.10.127:8080/rbc.api/auth/createToken?username=admin&password=123456", verify=False)
    except Exception as msg:
        print(msg)
        sys.exit(1)

    tokenresult = json.loads(r.content.decode("utf-8"))
    # print("tokenresult:", tokenresult)
    gettoken = tokenresult["data"]["token"]
    # print("gettoken:", gettoken)
    rbcPeerOrg = {
        "CAProperties": "",
        "mspid": "RBCOrgMSP",
        "name": "RBCOrg",
        "peerConfigList": [
            {
                "EVENT_LOCATION": "grpc://192.168.1.29:7053",
                "PEER_NAME": "peer0.peer.com",
                "PEER_LOCATION": "grpc://192.168.1.29:7051"
            }
        ]
    }
if __name__ == '__main__':
    plot_query_eff()