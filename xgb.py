from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = load_boston()
#波士顿数据集非常简单，但它所涉及到的问题却很多
# print(data)

X = data.data
y = data.target

Xtrain,Xtest,Ytrain,Ytest = TTS(X,y,test_size=0.3,random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain,Ytrain) #训练
reg.predict(Xtest)
print(reg.score(Xtest,Ytest))
print(y.mean())
print(MSE(Ytest,reg.predict(Xtest)))

# reg.feature_importances_
reg = XGBR(n_estimators=100)
print(CVS(reg,Xtrain,Ytrain,cv=5).mean())

#来查看一下sklearn中所有的模型评估指标
import sklearn
print(sorted(sklearn.metrics.SCORERS.keys()))

#使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=100)
print(CVS(rfr,Xtrain,Ytrain,cv=5).mean())#0.7975497480638329

lr = LinearR()
print(CVS(lr,Xtrain,Ytrain,cv=5).mean())#0.6835070597278085