import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#1.读取数据集
dataset = pd.read_csv('../dataset/iris.data')

#添加列名
column_names = ['sepal length','sepal width','petal length','petal width','class']
dataset.columns = column_names

#通过head()函数,来检查数据的前几行,确保苏韩剧能够加载正确
print(dataset.head())

#将前4列划分为特征X,最后一列class就是目标y
X = dataset.drop('class',axis=1)
y = dataset['y']
#划分训练集与测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)

#使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
