# encoding utf-8
import numpy as np
import os
from knn_dating import *

# 将图像数据处理成1x1024的向量
def img2vect(filename):
    vect = np.zeros((1,1024))
    fr = open(filename, 'r')
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            vect[0,i*32+j] = int(linestr[j])
    return vect

def hwtest():
    train_simple_list = os.listdir("trainingDigits")
    m_train = len(train_simple_list)
    train_dataset = np.zeros((m_train,1024))
    class_tag = []
    for i in range(m_train):
        filename = train_simple_list[i]
        train_dataset[i,:] = img2vect("trainingDigits/%s" %(filename))
        class_tag.append(int(filename.split('.')[0].split('_')[0]))
    error_count = 0
    error_rate = 0.0
    test_simple_list = os.listdir("testDigits")
    m_test = len(test_simple_list)
    for item in test_simple_list:
        test_vect = img2vect("trainingDigits/%s" %(item))
        class_tag_test = int(item.split('.')[0].split('_')[0])
        result_class = classify0(test_vect,train_dataset,class_tag,3)
        print("the classify result is %d, the answer is %d" %(result_class,class_tag_test))
        if result_class != class_tag_test:
            error_count+=1
    print("the error rate is %f." %(error_count/m_test))

if __name__=='__main__':
    hwtest()