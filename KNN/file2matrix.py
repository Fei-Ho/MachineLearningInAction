# encoding utf-8
import numpy as np


def file2matrix(filename):
    fr = open(filename, 'r')
    arryoflines = fr.readlines()
    numoflines = len(arryoflines)
    retMat = np.zeros((numoflines, 3))
    classLabelVector = []
    index = 0
    for line in arryoflines:
        line = line.strip()
        items = line.split('\t')
        retMat[index, :] = items[0: 3]
        classLabelVector.append(int(items[-1]))
        index += 1
    return retMat, classLabelVector
