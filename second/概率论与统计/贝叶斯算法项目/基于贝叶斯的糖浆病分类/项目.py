import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. 读取糖尿病数据集
dataset = pd.read_csv('./dataset/diabetes_data_upload.csv')

# 查看数据集基本信息，确认加载正确
print("数据集形状:", dataset.shape)
print("\n数据集前5行:")
print(dataset.head())
print("\n数据集列名:", dataset.columns.tolist())

# 2. 分离特征和目标变量
X = dataset.drop('class', axis=1)  # 特征（16个症状/基本信息）
y = dataset['class']  # 目标变量（是否患病）

# 3. 处理类别特征：将Yes/No等分类特征编码为数值
# 对所有特征列进行独热编码
X_encoded = pd.get_dummies(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, train_size=0.5, random_state=42
)

# 5. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 初始化并训练高斯朴素贝叶斯模型
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)

# 7. 模型预测
y_pred = nb_classifier.predict(X_test_scaled)

# 8. 模型评估
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'\n模型准确率 (Accuracy): {accuracy:.4f}')
print(f'\n混淆矩阵 (Confusion Matrix):\n{conf_matrix}')
print(f'\n分类报告 (Classification Report):\n{class_report}')

# 9. t-SNE降维可视化
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(X_test_scaled)

# 将标签转为数值（1→0，2→1），方便绘图
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_test)

# 绘制散点图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    x_tsne[:, 0], x_tsne[:, 1],
    c=y_test_numeric, cmap='coolwarm', s=60, alpha=0.8
)
plt.title('t-SNE Visualization of Diabetes Prediction Results')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
# 设置图例，对应原始类别
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=['Diabetic', 'Non-Diabetic'],
    title='Classes'
)
plt.grid(alpha=0.3)
plt.show()