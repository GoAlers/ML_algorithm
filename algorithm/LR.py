# -*- coding: utf-8 -*-

# 公式：
# 预测函数


from numpy import *
import matplotlib.pyplot as plt


# def loadDataSet():
#     dataMat = []
#     labelMat = []
#     fr = open('testlr.txt')
#     # 逐行读入数据，然后strip去头去尾，用split分组
#     for line in fr.readlines():
#         lineArr = line.strip().split('   ')
#         dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
#         labelMat.append(int(lineArr[2]))
#     return dataMat,labelMat

LINE_OF_DATA = 12
# LINE_OF_TEST = 6

#载入自定义数据
def createTrainDataSet():
	trainDataMat = [[1, 1, 4],
					[1, 2, 3],
					[1, -2, 3],
					[1, -2, 2],
					[1, 0, 1],
					[1, 1, 2]]
	trainShares = [1, 1, 1, 0, 0, 0]
	return trainDataMat, trainShares

# def createTestDataSet():
# 	testDataMat = [[1, 1, 1],
# 				   [1, 2, 0],
# 				   [1, 2, 4],
# 				   [1, 1, 3],
# 				   [1, 3, 4],
# 				   [1, 2, 3]]
# 	return testDataMat

# 定义sigmoid函数
def sigmoid(z):
	return 1.0 / (1 + exp(-z))

# 定义梯度函数，传入数据集、label;步长、循环次数
def gradient(dataMatIn, classLabels, alpha, maxCycles):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()    #y值
	m, n = shape(dataMatrix)                   #矩阵行列数
	weights = ones((n, 1))						#初试回归系数1
	for i in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

def plot(weights):
	dataMat, labelMat = createTrainDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	x1,y1,x2,y2 = [],[],[],[]
	for i in range(n):
		if int(labelMat[i]) == 1:
			x1.append(dataArr[i, 1])
			y1.append(dataArr[i, 2])
		else:
			x2.append(dataArr[i, 1])
			y2.append(dataArr[i, 2])
	fig = plt.figure(figsize=(15,8))
	ax = fig.add_subplot(111)
	ax.scatter(x1, y1,c='blue', marker='s')
	ax.scatter(x2, y2, c='green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.show()

def classifyVector(inX, weights):
	prob = sigmoid(sum(inX * weights))
	if prob > 0.5:
		return 1
	else:
		return 0

# def classifyAll(dataSet, weights):
# 	predict = []
# 	for vector in dataSet:
# 		predict.append(classifyVector(vector, weights))
# 	return predict

# def main():
# 	trainDataSet, trainShares = createTrainDataSet()
# 	# testDataSet = createTestDataSet()
# 	regMatrix = gradient(trainDataSet, trainShares, 0.01, 600)
# 	print("regMatrix = \n", regMatrix)
# 	plot(regMatrix.getA())
# 	# predictShares = classifyAll(testDataSet, regMatrix)
# 	# print("predictResult: \n", predictShares)

if __name__ == '__main__':
	trainDataSet, trainShares = createTrainDataSet()
	regMatrix = gradient(trainDataSet, trainShares, 0.01, 600)
	print("regMatrix = \n", regMatrix)
	plot(regMatrix.getA())