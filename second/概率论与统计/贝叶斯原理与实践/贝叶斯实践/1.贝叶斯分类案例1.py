''''''
'''
大致流程:
散点输入 -> 先验概率、条件概率->后验概率->显示决策边界

P(A|B)=P(A)P(B|A)/P(B)
'''

'''
协方差矩阵计算
cov()
'''
import numpy as np

X = np.array([[0,2],[1,1],[2,0]])
"""
0  2
1  1
2  0 
"""
#计算协方差矩阵时,输入的是:特征*样本 2*3矩阵
"""
0  1  2
2  1  0
"""
X = X.T

#np.cov()
#结果:2*2矩阵
print('numpy的结果',np.cov(X))

#假设每一行代表一个特征 x1, x2


#手动计算协方差矩阵
mean_X1 = 1
mean_X2 = 1

'''
X1X1  X1X2

X2X1  X2X2
'''
cov_X1X1 = np.sum((X[0,:]-mean_X1)**2)/(X.shape[1] - 1)
cov_X1X2 = np.sum((X[0,:]-mean_X1) * (X[1,:] - mean_X2)) / (X.shape[1]-1)
cov_X2X1 = cov_X1X2
covX2X2 = np.sum((X[1,:]-mean_X2) ** 2) / (X.shape[1] - 1)

print("自己计算出的结果",np.array([[cov_X1X1,cov_X1X2],[cov_X2X1,covX2X2]]))
