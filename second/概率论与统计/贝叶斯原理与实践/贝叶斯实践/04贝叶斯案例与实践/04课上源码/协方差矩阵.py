import numpy as np

# 3个样本，2个特征，3*2矩阵
X = np.array([[0, 2], [1, 1], [2, 0]])
"""
0  2
1  1
2  0
"""
# 计算协方差矩阵时，输入是：特征*样本  2*3矩阵
"""
0  1  2
2  1  0
"""
X = X.T

# 结果：2*2矩阵
print("numpy的cov结果：", np.cov(X))

# 假设每一行代表一个特征  x1  x2


# 手动计算协方差矩阵

# 计算平均值
mean_X1 = 1
mean_X2 = 1

"""
X1X1  X1X2
X2X1  X2X2
"""

cov_X1X1 = np.sum((X[0, :] - mean_X1) ** 2) / (X.shape[1] - 1)
cov_X1X2 = np.sum((X[0, :] - mean_X1) * (X[1, :] - mean_X2)) / (X.shape[1] - 1)
cov_X2X1 = cov_X1X2
cov_X2X2 = np.sum((X[1, :] - mean_X2) ** 2) / (X.shape[1] - 1)

print("自己计算的结果：", np.array([[cov_X1X1, cov_X1X2], [cov_X2X1, cov_X2X2]]))














