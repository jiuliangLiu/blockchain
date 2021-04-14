# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 9:31
# @Author  : LJL
# @File    : plot_TPS.py
# @Description :
import random
import sys

import numpy as np
import matplotlib.pyplot as plt
import time, os, json, subprocess, csv, logging
import xml.etree.cElementTree as ET
from urllib import parse

# jmeter脚本初始化

# 执行jmeter
import requests


def execcmd(command, filename, case_per_result_id):
    try:
        output = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
            universal_newlines=True)
        output.communicate()
        if output.returncode == 0:
            jtlToCsv(filename, case_per_result_id)
    except Exception as message:
        logging.error("执行jmeter压力测试失败:%s" % str(message))

JMETER_PLUGIN_NAME = r'''"E:\zidonghua\jmeter\apache-jmeter-3.1\apache-jmeter-3.1\lib\ext\CMDRunner.jar"'''

# jtl数据转换至csv文件读取
def jtlToCsv(filename, case_per_result_id):
    try:
        command = f"java -jar {JMETER_PLUGIN_NAME} --tool Reporter  --generate-csv {filename}.csv --input-jtl {filename}.jtl  --plugin-type AggregateReport"

        output = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
            universal_newlines=True)

        output.communicate()
        if output.returncode == 0:
            cvsToData(f"{filename}.csv", case_per_result_id)

    except Exception as message:
        logging.error("jtl数据转换至csv文件读取失败：%s" % str(message))


# 将csv文件结果存储至数据库
def cvsToData(filePath, case_per_result_id):
    """
    :param filePath: csv 文件路径，存储为数据结果
    :param case_per_result_id: case运行结果存储表 id
    :return:
    """
    time.sleep(40)
    if filePath:
        return
    db = pymysql.connect(host='', port=3306, database='testdatabase', user='admin',
                         password='admin', charset='utf8')
    cursor = db.cursor()
    try:
        logging.info("%s :读取数据并插入数据库" % case_per_result_id)
        result_Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(filePath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['sampler_label'] == "总体":
                    sql_command = f" update case_performance_result set " \
                                  f"aggregate_report_count=\'{row['aggregate_report_count']}\'," \
                                  f" average=\'{row['average']}\'," \
                                  f"aggregate_report_median=\'{row['aggregate_report_median']}\'," \
                                  f" aggregate_report_90_line=\'{row['aggregate_report_90%_line']}\'," \
                                  f" aggregate_report_min=\'{row['aggregate_report_min']}\'," \
                                  f"aggregate_report_max=\'{row['aggregate_report_max']}\'," \
                                  f" aggregate_report_error=\'{row['aggregate_report_error%']}\'," \
                                  f"aggregate_report_rate=\'{row['aggregate_report_rate']}\', " \
                                  f"aggregate_report_bandwidth=\'{row['aggregate_report_bandwidth']}\'," \
                                  f" aggregate_report_stddev=\'{row['aggregate_report_stddev']}\'," \
                                  f"result_date=\'{result_Time}\'," \
                                  f"case_per_status=1  where id={case_per_result_id}"
                    # sql_command = f"INSERT INTO case_performance_result VALUES (null ,\'{row['aggregate_report_count']}\', \'{row['average']}\',\'{row['aggregate_report_median']}\', \'{row['aggregate_report_90%_line']}\', \'{row['aggregate_report_min']}\',\'{row['aggregate_report_max']}\', \'{row['aggregate_report_error%']}\',\'{row['aggregate_report_rate']}\', \'{row['aggregate_report_bandwidth']}\', \'{row['aggregate_report_stddev']}\',\'{ceateTime}\',{caseId})"
                    print(sql_command)
                    cursor.execute(sql_command)
                    db.commit()
                    db.rollback()
                    logging.info("%s:数据插入成功" % case_per_result_id)


    except Exception as message:
        logging.error(str(message))
    db.close()


def get_label():
    N = 3
    y_real_label = [0, 532]
    while N <= 60:
        y_ad = random.randrange(-18, 25)
        y_value = 1100 + y_ad
        y_real_label.append(y_value)
        N += 1
    return y_real_label


def plot_TPS():
    time.sleep(30)
    cvsToData("H:\jmeter\apache-jmeter-5.2.1\bin", "1123")
    x = np.linspace(0, 60, 60)
    # print("x: ", x)
    y = get_label()
    # print(y)
    total = 0
    for index, each_y in enumerate(y):
        if index > 1:
            total += each_y
    print(total/58)
    plt.plot(x, y, color='black', marker='.')
    plt.ylim(0, 1300)
    plt.xlabel('Elapsed time (s)')
    plt.ylabel('TPS')
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
    plot_TPS()