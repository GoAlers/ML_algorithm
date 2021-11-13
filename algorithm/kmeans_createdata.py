# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

# 创建数据集
def createDataSet():

    dataSet = [[0.0, 1.0],
			   [0.0, 2.0],
               [1.0, 1.5],
			   [1.5, 0.0],
               [2.5, 0.0],
               [3.5, 0.5],
			   [4.0, 0.0],
			   [6.0, 2.0],
               [5.0, 2.5]]
    return dataSet

# 欧式距离公式
def Ecluddist(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 构建聚簇中心，取k个随机质心centroids
def randCent(dataSet, k):
	"""
	输入：数据集, 聚类个数
	输出：k个随机质心的矩阵
	"""
	n = shape(dataSet)[1]
	centroids = mat(zeros((k, n)))   # 每个质心有n个坐标值，总共要k个质心
	for j in range(n):
		maxJ = max(dataSet[:, j])
		minJ = min(dataSet[:, j])
		rangeJ = float(maxJ - minJ)  #获取每个特征的范围
		centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  #随机生成k*1的随机数矩阵
	return centroids

def kMeans(dataSet, k, distMeans =Ecluddist, createCent=randCent):
	"""
	输入：数据集, 聚类个数, 距离计算函数, 生成随机质心函数
	输出：质心矩阵, 簇分配和距离矩阵
	"""
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))   #m行的矩阵
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:     # 判断聚类是否收敛
		clusterChanged = False
		for i in range(m): 	#把每一个数据点划分到离它最近的中心点
			INF = 100.0
			minDist = INF;minIndex = -1  # 初始化最小值
			for j in range(k):
				distJI = distMeans(centroids[j, :], dataSet[i, :])  #计算一组数据与质心之间的距离
				if distJI < minDist:   # 如果第i个数据点到第j个中心点更近，则将i归属为j
					minDist = distJI
					minIndex = j
			if clusterAssment[i, 0] != minIndex:	# 如发生变化，则继续迭代
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist**2   # 并将第i个数据点的分配情况存入字典
		# print(centroids)

		for cent in range(k): #	 # 重新计算中心点
			ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
			centroids[cent, :] = mean(ptsInClust, axis=0)  #均值求数据的中心点
	return centroids, clusterAssment


if __name__ == '__main__':
	dataSet = createDataSet()
	datamat = mat(dataSet)
	resultCent, clustAssing = kMeans(datamat,3)
	print(resultCent)



