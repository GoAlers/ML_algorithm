import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 分类的数据
x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
# plt.scatter(x[:,0], x[:,1])
# print(x)
# 分为两组
clf = KMeans(n_clusters=2)
clf.fit(x)  # 分组

center = clf.cluster_centers_  # 两组数据点的中心点
labels = clf.labels_  # 每个数据点所属分组
print(center)
print(labels)

for i in range(len(labels)):
    plt.scatter(x[i][0], x[i][1], c=('r' if labels[i] == 0 else 'b'))
plt.scatter(center[:, 0], center[:, 1], marker='*', s=100)

# 预测
predict = [[2, 1], [6, 9]]
label = clf.predict(predict)
for i in range(len(label)):
    plt.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')

plt.show()