import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np
#读取数据集
dataset = pd.read_csv('./dataset/iris.data')
#添加列名
column_names = ['sepal length','sepal width','petal length','petal width','class']
dataset.columns = column_names
# print(dataset.head())
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
#预测
y_pred = knn.predict(X_test)

#评估
accuracy = accuracy_score(y_test,y_pred)
conf_mat = confusion_matrix(y_test,y_pred)

print(f"测试集准确率:{accuracy:.4f}")
