# encoding utf-8
import matplotlib.pyplot as plt
import numpy as np
from file2matrix import *


def classify0(inVec, classVec, labels, k):
    m = classVec.shape[0]
    dupInVec = np.tile(inVec, (m, 1))
    diffMat = dupInVec - classVec
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    y = distance.argsort()
    countClass = { }
    for i in range(k):
        label = labels[y[i]]
        countClass[label] = countClass.get(label, 0) + 1
    sortedCountclass = sorted(countClass.items(),key=lambda item: item[1], reverse=False)
    return sortedCountclass[0][0]

def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    m = dataset.shape[0]
    normdataset = (dataset - np.tile(minVals, (m, 1)))/np.tile(ranges, (m, 1))
    return normdataset, minVals, ranges

def testAccuracy():
    # 读入数据
    dataset_input,labels = file2matrix("datingTestSet2.txt")
    # 数据标准化
    dataset,minVals,ranges = autoNorm(dataset_input)
    # 测试集占比
    horate = 0.10
    m = dataset.shape[0]
    test_num = int(m * 0.1)
    error_count=0
    error_rate=0.0
    for i in range(test_num):
        classify_result = classify0(dataset[i,:], dataset[test_num:m,:], labels[test_num:m], 3)
        print("The classify result is %d, the answer is %d" %(classify_result,labels[i]))
        if classify_result != labels[i]:
            error_count+=1
    print("The total error rate is %f" %(error_count/m))

if __name__=='__main__':
    # 读取文本数据并转换成矩阵测试
    dataDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(dataDataMat, "\n", datingLabels)
    # 画散点图测试
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataDataMat[:, 0], dataDataMat[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()
    # 数据标准化测试
    normMat, minVals, ranges = autoNorm(dataDataMat)
    print(normMat, '\n', minVals, '\n', ranges)
    # test函数测试
    accuracy = testAccuracy()
