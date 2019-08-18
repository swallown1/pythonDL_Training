import os
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    fname = os.path.join('./data', 'jena_climate_2009_2016.csv')
    f = open(fname)
    data = f.read()
    f.close()
    lines = data.split('\n')
    # 每个表的表头
    header = lines[0].split(',')
    # 表的数据  420551条
    lines = lines[1:]
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values
    return float_data

def test():
    data = get_data()
    lines = data.split('\n')
    # 每个表的表头
    header = lines[0].split(',')
    # 表的数据  420551条
    lines = lines[1:]
    # print(lines)
    # print(header)
    # 将表数据转化成numpy的数组
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    # 绘制温度时间序列
    temp = float_data[:, 1]  # 第一列是温度时间序列
    # plt.plot(range(len(temp)),temp,'b')
    # plt.show()

    # 绘制前十天的温度时间序列
    plt.plot(range(1440), temp[:1440])
    plt.show()

# test()
