import numpy as np
import matplotlib.pyplot as plt


def pdf(x, mean, cov):
    n = len(mean)
    # 计算PDF的系数
    coeff = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(np.linalg.det(cov)))
    # 计算PDF的指数部分
    exponet = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponet)

    # 1. 散点输入
class1_points = np.array([[1.9, 1.2], [1.5, 2.1], [1.9, 0.5], [1.5, 0.9], [0.9, 1.2], [1.1, 1.7], [1.4, 1.1]])
class2_points = np.array([[3.2, 3.2], [3.7, 2.9], [3.2, 2.6], [1.7, 3.3], [3.4, 2.6], [4.1, 2.3], [3.0, 2.9]])
class3_points = np.array([[3.3, 1.2], [3.8, 0.9], [3.3, 0.6], [2.8, 1.3], [3.5, 0.6], [4.2, 0.3], [3.1, 0.9]])

X = np.concatenate((class1_points, class2_points, class3_points))
y = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points)), np.ones(len(class3_points)) + 1))  # 3个类别标签：0,1,2

# 2. 计算先验概率（修正：包含3个类别）
prior_probabilities = [
    np.sum(y == 0) / len(y),  # 类别0的先验
    np.sum(y == 1) / len(y),  # 类别1的先验
    np.sum(y == 2) / len(y)   # 类别2的先验（补充）
]
print("先验概率：", prior_probabilities)

# 3. 计算均值和协方差（3个类别）
class_means = [
    np.mean(X[y == 0], axis=0),
    np.mean(X[y == 1], axis=0),
    np.mean(X[y == 2], axis=0)
]

class_covs = [
    np.cov(X[y == 0].T),
    np.cov(X[y == 1].T),
    np.cov(X[y == 2].T)
]

# 生成网格点用于绘制决策边界
xx, yy = np.meshgrid(np.arange(0, 5, 0.05), np.arange(0, 4, 0.05))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 4. 计算每个网格点的后验概率并分类
grid_label = []
for point in grid_points:
    posterior = []
    for i in range(3):  # 循环3个类别
        likelihood = pdf(point, class_means[i], class_covs[i])  # 条件概率
        posterior.append(prior_probabilities[i] * likelihood)  # 后验概率（简化版，忽略证据因子）
    grid_label.append(np.argmax(posterior))  # 取后验最大的类别

grid_label = np.array(grid_label).reshape(xx.shape)

# 5. 绘图
plt.scatter(class1_points[:,0], class1_points[:,1], c='blue', label="class 1")
plt.scatter(class2_points[:,0], class2_points[:,1], c='red', label="class 2")
plt.scatter(class3_points[:,0], class3_points[:,1], c='black', label="class 3")  # 修正：原代码误写为class2_points

# 绘制决策边界（3类的决策边界用levels控制）
plt.contour(xx, yy, grid_label, levels=[0.5, 1.5], colors='green')  # 0-1和1-2的边界

plt.legend()
plt.show()