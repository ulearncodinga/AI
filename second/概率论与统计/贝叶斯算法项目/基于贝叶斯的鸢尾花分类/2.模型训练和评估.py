import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#1.读取数据集
dataset = pd.read_csv('../dataset/iris.data')

#添加列名
column_names = ['sepal length','sepal width','petal length','petal width','class']
dataset.columns = column_names

#通过head()函数,来检查数据的前几行,确保苏韩剧能够加载正确
print(dataset.head())

#将前4列划分为特征X,最后一列class就是目标y
X = dataset.drop('class',axis=1)
y = dataset['class']
#划分训练集与测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)

#使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#初始化朴素贝叶斯分类器
nb_classifier = GaussianNB()

#根据划分好的特征取训练模型
nb_classifier.fit(X_train,y_train)

#在测试及上进行预测
y_pred = nb_classifier.predict(X_test)

#准确率:计算分类模型的准确率,可以计算准确个数
accuracy = accracy_score(y_test,y_pred)

#混淆矩阵:可以直观地展示分类模型在各个类别上的预测情况
#返回一个二维数组(也就是混淆矩阵),矩阵的行表示真实类别,矩阵的列表示预测情况
conf_matrix = confusion_matrix(y_test,y_pred)
#精准率:指该类别被正确预测的样本数(真正例)与所有被预测为该类别的样本数(假正例). 真正例/(真正例+假正例)
#召回率:指该类别被正确预测的样本数(真正例)与所有实际属于该类别的样本数(假反例)
#f1值:精准率与召回率的调和平均数: 2 * pre * recall / (pre + recall)
#支持度:指每个类别在真实标签中出现的样本数量
#宏平均:对各个类别的指标(pre,recall,f1)的简单平均
#加权平均:根据每个类别的支持度对每个类别的指标进行加权平均
class_report = classification_report(y_test,y_pred)
print(f'Accracy:{accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classfication Report:\n{class_report}')
