# encoding utf-8

from math import log
from ploter import *

# 计算香农信息熵
def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts={}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

# 数据生成
def createDataSet():
    dataSet = [[1,1,'Yes'],
               [1, 1, 'Yes'],
               [1, 0, 'No'],
               [0, 1, 'No'],
               [0, 1, 'No']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

def splitData(dataset,axis,value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis]==value:
            retfeatvec = featVec[:axis]
            retfeatvec.extend(featVec[axis+1:])
            retDataSet.append(retfeatvec)
    return retDataSet

# 计算每个特征的信息增益，选择最佳的划分特征
def chooseBestFeaturesToSplit(dataset):
    baseEnt = calcShannonEnt(dataset) # 原始整个数据集的信息熵
    numFeatiures = len(dataset[0])-1 # 一个样本的特征数，最后一个是标签
    bestInfoGain = 0.0 # 最大的信息增益
    bestFeature = -1 # 最佳的特征下标
    for i in range(numFeatiures):
        valueoffeature = [ example[i] for example in dataset]
        uniquefeature = set(valueoffeature) # 第i个特征的取值集合
        tempEnt = 0.0
        for value in uniquefeature:
            aftersplit = splitData(dataset,i,value)
            prob = len(aftersplit)/float(len(dataset))
            tempEnt += prob*calcShannonEnt(aftersplit)
        if baseEnt-tempEnt>bestInfoGain:
            bestInfoGain = baseEnt-tempEnt
            bestFeature = i
    return bestFeature

# 返回一个classList字典中value最大的key
def majorityCnt(classList):
    classCounts = {}
    for vote in classList:
        if vote not in classCounts.keys():
            classCounts[vote]=1
        else:
            classCounts[vote]+=1
    sortList = sorted(classCounts.items(), key=lambda item: item[1], reverse=True)
    return sortList[0][0]

# 生成决策树
def createTree(dataset,labels):
    # 数据集所有样本的标签列表
    classList = [ example[-1] for example in dataset]
    # 数据集中样本都为同一类，停止
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 所有的特征都被划分过了，停止
    if len(dataset[0])==1:
        return majorityCnt(classList)
    # 递归构建决策树
    bestFeature = chooseBestFeaturesToSplit(dataset)
    bestFeatureLabel = labels[bestFeature]
    del (labels[bestFeature])
    mytree = {bestFeatureLabel:{}}
    valueList = [ example[bestFeature] for example in dataset]
    uniqvalue = set(valueList)
    for value in uniqvalue:
        sublabels = labels[:]
        mytree[bestFeatureLabel][value]=createTree(splitData(dataset,bestFeature,value),sublabels)
    return mytree

# 使用分类器
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

# 使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__=='__main__':
    # majorityCent()函数测试
    # classList = ['Yes','No','Maybe','Yes','No','No']
    # print(majorityCnt(classList))

    # 生成数据集
    dataset,labels = createDataSet()
    labels_copy = labels.copy() # labels原始数据将在建树的过程中被损毁，备份
    print("数据集:", dataset)

    # 计算信息熵
    # print("信息熵:%f" % calcShannonEnt(dataset))
    # 混合的数据越多，熵越高，额外增加一个maybe的分类
    # dataset[0][-1] = 'maybe'
    # print("增加maybe类别后的信息熵:%f" % calcShannonEnt(dataset),"混合的数据越多，熵越高(数据的无序性)")

    # splitData测试
    # print("划分出第一个特征值为1的样本",splitData(dataset,0,1))
    # 选择最佳划分特征测试
    # print("最佳的划分特征是第",chooseBestFeaturesToSplit(dataset),"个特征")

    # 生成决策树
    myTree = createTree(dataset, labels)
    print("生成的决策树：",myTree)
    # 画决策树
    createPlot(myTree)
    # 持久化决策树
    storeTree(myTree,'classifierStorage.txt')
    myTree = grabTree('classifierStorage.txt')
    print("从持久化层读取的决策树：", myTree)
    # 决策树的使用
    print(classify(myTree, labels_copy, [1, 0]))
    print(classify(myTree, labels_copy, [1, 1]))


