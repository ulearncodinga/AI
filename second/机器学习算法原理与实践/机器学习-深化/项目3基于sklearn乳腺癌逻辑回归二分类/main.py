import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_curve,auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#假设数据保存在breast-cancer.data文件中
dataset = pd.read_csv('./dataset/breast-cancer.data')

#添加列名
column_names = ["Class",'age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
dataset.columns = column_names
#检查前几行数据,确保加载正确
print(dataset.head())
#去除数据中的异常值
dataset = dataset.replace('?',np.nan)
dataset = dataset.dropna(axis=0)

#特征及,排除目标变量'Class'
X = dataset.drop(['Class'],axis=1)
y = dataset['Class']

#处理类别特征,使用独热编码
X_encoded = pd.get_dummies(X)
print(X_encoded.keys())

#划分训练集和测试集
X_train,X_test,y_train,y_test =  train_test_split(X_encoded,y,test_size=0.2,random_state=42)
#使用标准化进行特征缩放(本例中的特征都是离散型的实际上可以不进行归一化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#将字符串标签转换为数值标签,方便画图
label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train)
y_test_numeric = label_encoder.transform(y_test)

#训练逻辑回归模型
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train_scaled,y_train_numeric)
#预测正类别(类别为1)的概率
prepro = logreg.predict_proba(X_test_scaled)[:,1]

#计算假正比例(fpr),真正例率(tpr)和阈值
fpr,tpr,thresholds = roc_curve(y_test_numeric,prepro)
#计算ROC曲线下的面积(AUC)
roc_auc = auc(fpr,tpr)

#绘制ROC曲线
plt.figure(figsize=(10,6))
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc='lower right')
plt.show()