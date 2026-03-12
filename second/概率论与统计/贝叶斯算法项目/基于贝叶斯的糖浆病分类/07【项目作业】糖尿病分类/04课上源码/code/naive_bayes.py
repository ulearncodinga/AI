import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 假设数据保存在diabetes_data_upload.csv文件中
dataset = pd.read_csv('../dataset/diabetes_data_upload.csv', sep=',')

# 检查数据的前几行，确保数据加载正确
print(dataset.head())

# 假设特征列为X，目标变量列为y
X = dataset.drop('class', axis=1)
y = dataset['class']

# 处理类别特征：使用独热编码
categorical_features = X.keys()[0:]
X_encoded = pd.get_dummies(X, columns=categorical_features)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=42)

# 使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化朴素贝叶斯分类器
nb_classifier = GaussianNB()

# 在训练集上训练模型
nb_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = nb_classifier.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_test_scaled)

# 将字符串标签转换为数值标签，方便之后画图
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_pred)

# 创建散点图进行可视化
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test_numeric, cmap='viridis')
plt.title('t-SNE Visualization of naive_bayes Predictions')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()
