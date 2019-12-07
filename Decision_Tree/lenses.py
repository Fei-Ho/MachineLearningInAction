# encoding utf-8
import tree
import ploter
fr = open('lenses.txt')
lenses = [ inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
# 构造决策树
lensesTree = tree.createTree(lenses,lensesLabels)
# 打印构造的决策树
print(lensesTree)
# 图形化树
ploter.createPlot(lensesTree)