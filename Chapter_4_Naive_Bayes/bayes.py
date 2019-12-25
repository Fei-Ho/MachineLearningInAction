# encoding utf-8
import numpy as np

def loadDataSet():
    # 每一个post按word切分的结果
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，标记上面的每条post是否是侮辱性的，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本进行词条切分后的文档集合、类别标签向量
    return postingList, classVec

# 得到所有post中词汇的集合，即词汇表
def createVocaList(dataset):
    vocabulary_set = set([])
    for document in dataset:
        vocabulary_set = vocabulary_set|set(document)
    return list(vocabulary_set)

# 输入某个post和词汇表，输出词向量
# 词汇表中的单词在该条post中有出现，则为1，否则为0
# 词集模型，只统计每个词是否出现，而不统计其出现次数
def word2vec(input_set,vocabulary_list):
    vect = [0]*len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            vect[vocabulary_list.index(word)]=1
        else:
            print("The word %s is not in my vocabulary" %(word))
    return vect

# 词袋模型，统计每个词出现的次数
def bagofword2vec(input_set,vocabulary_list):
    vect = [0]*len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            vect[vocabulary_list.index(word)] += 1
        else:
            print("The word %s is not in my vocabulary" %(word))
    return vect

# p(ci|w) = (p(w|ci)p(ci)) / p(w)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 即，p(c1)，p(c0)=1-p(c1)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Demon = 2.0; p1Demon = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Demon += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Demon += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Demon) # 所有侮辱性样本中,每个词的出现概率
    p0Vect = np.log(p0Num/p0Demon) # 所有非侮辱性样本中,每个词的出现概率
    return p0Vect,p1Vect,pAbusive

def classifyNB(vect2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vect2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vect2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    # 加载数据集
    listOPost,listClasses = loadDataSet()
    myVocabList = createVocaList(listOPost)
    # 训练分类器
    trainMat = []
    for postinDoc in listOPost:
        trainMat.append(word2vec(postinDoc,myVocabList))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
    # 非侮辱性样本测试
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(word2vec(testEntry,myVocabList))
    print(testEntry,"classified as :",classifyNB(thisDoc,p0V,p1V,pAb))
    # 侮辱性样本测试
    testEntry=['stupid','garbage']
    thisDoc=np.array(word2vec(testEntry,myVocabList))
    print(testEntry, "classified as :", classifyNB(thisDoc,p0V,p1V,pAb))

if __name__=='__main__':
    post_list,class_vec = loadDataSet() #加载训练数据
    vocabulary_list = createVocaList(post_list) # 生成词汇表
    print("词汇表：\n",vocabulary_list)
    print("词向量示例：\n",word2vec(post_list[3],vocabulary_list))
    trainMatrix = []
    for item in post_list:
        trainMatrix.append(word2vec(item,vocabulary_list))
    p0Vect,p1Vect,pAbusive = trainNB0(trainMatrix,class_vec)
    print("p0Vect\n",p0Vect,"\np1Vect\n",p1Vect,"\npAbusive\n",pAbusive)
    # 测试分类器
    testingNB()