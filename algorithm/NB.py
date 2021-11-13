# -*- coding: utf-8 -*-

from numpy import *
import operator
import time

SIZE_OF_DATA = 5
SIZE_OF_TEST = 5

def read_input(filename):
	with open(filename,encoding='utf-8') as fr:
		corpus = []
		for text in fr.readlines()[1:]:
			for word in text.strip().split('\t')[1].split():
				corpus.append(word)
		allwords = set(corpus)

	matN = len(allwords)
	returnMat = zeros((SIZE_OF_DATA + SIZE_OF_TEST, matN))
	shares = []
	index = 0
	with open(filename,encoding='utf-8') as fr:
		for line in fr.readlines()[1:]:
			setFromLine = set(line.strip().split('\t')[1].split())
			oneLine = []
			for s in allwords:
				if s in setFromLine:
					oneLine.append(1)
				else:
					oneLine.append(0)
			returnMat[index, :] = oneLine
			if index < SIZE_OF_DATA:
				shares.append(float(line.strip().split('\t')[-1].strip()))
			index += 1
	return returnMat[:SIZE_OF_DATA,:], returnMat[SIZE_OF_DATA:,:], shares

def norm(inputMat):
	outputMat = inputMat.copy()
	m, n = shape(inputMat)
	for i in range(m):
		lineSum = sum(inputMat[i, :])
		for j in range(n):
			outputMat[i, j] = inputMat[i, j] / lineSum
	return outputMat

def cosineFunction(a, b):
	l = len(a)
	up = 0
	for i in range(l):
		up += a[i] * b[i]
	down1 = linalg.norm(a)
	down2 = linalg.norm(b)
	return (up / (down1 * down2))

def classify(trainDataSet, testDataSet, dataShares):
	trainDataSet = trainDataSet.transpose()
	emotionMat = dot(trainDataSet, dataShares) # 第i个词和情感的相关度
	count = sum(trainDataSet)
	for i, word in enumerate(emotionMat):
		emotionMat[i] = word * sum(trainDataSet[i]) / count
		# 由词推断出情感的概率 =
		#					当前文本已知情感出现词的概率
		#				  * 当前训练文本中的情感概率值
		#				  / 所有文本中出现词的概率
	predictShares = dot(testDataSet, emotionMat)
	return norm(mat(predictShares))

if __name__ == '__main__':
	trainMat, testMat, shares = read_input('testNB')
	normTrainMat = norm(trainMat)
	normTestMat = norm(testMat)
	predictShares = classify(normTrainMat, normTestMat, shares)
	print(predictShares)
