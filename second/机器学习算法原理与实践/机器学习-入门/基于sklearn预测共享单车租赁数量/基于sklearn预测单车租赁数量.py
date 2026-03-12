import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

#读取数据
dataset = pd.read_csv('./day.csv',sep=',')
#打印前几行的数据确保加载正确
print(dataset.head())

#把没用的三列单独拿出来,把剩下的进行编码
X = dataset.drop(['cnt','instant','dteday'],axis=1)
y = dataset['cnt']

#定义一个包含所有类别特征列名的列表,这些通常包含离散的分类信息,如季节,年份
categorical_features = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
#使用pd.get_dummies函数对这些类别特征列进行独热编码,将每个类别特征转换为多个二进制的虚拟变量列
X_encoded = pd.get_dummies(X,columns=categorical_features)


#划分训练集和测试集
train_ratio = 0.8
X_train = X_encoded[:int(train_ratio * len(dataset))]
X_test = X_encoded[int(train_ratio*len(dataset)):]
y_train =y[:int(train_ratio*len(dataset))]
y_test = y[int(train_ratio*len(dataset)):]

#使用标准化数据进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#创建线性回归模型
linear_reg_model = LinearRegression()
#训练模型
linear_reg_model.fit(X_train_scaled,y_train)
# 进行预测
y_pred = linear_reg_model.predict(X_test_scaled)
#评估模型性能
#计算均方误差
mse = mean_squared_error(y_test,y_pred)
#计算R-squared值(决定系数)表示模型对目标变量变异的解释程度
r2 = r2_score(y_test,y_pred)
print(f'Mean Squared Error:{mse}')
print(f'R-squared:{r2}')

plt.figure(0)
plt.scatter(y_test,y_pred)
# plt.scatter(y_test,y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values in Linear Regression")

#根据测试机索引的排序后的排序,这里假设测试集索引对应着时间顺序或者其他有意义的顺序
sorted_indices = X_test.index.argsort()
#根据排序后的索引获取对应的实际值
y_test_sorted = y_test.iloc[sorted_indices]
#根据排序后的索引获取对应的预测值,并转换为pandas的Series类型
y_pred_sored = pd.Series(y_pred).iloc[sorted_indices]


#绘制第二个图像
plt.figure(1)
plt.plot(y_test_sorted.values,label='Actual Values',marker='o')
plt.plot(y_pred_sored.values,label="Predicted Values",marker='x')
plt.xlabel('Sample Index (sorted)')
plt.ylabel("Values")
plt.title("Actual vs Predicted in Linear Regression")
plt.legend()
plt.show()