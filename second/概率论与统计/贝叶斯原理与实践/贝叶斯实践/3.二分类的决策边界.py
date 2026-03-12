import numpy as np
import matplotlib.pyplot as plt



def pdf(x,mean,cov):
    n = len(mean)
    #计算PDF的系数
    coeff = 1 / ((2 * np.pi) ** (n/2) * np.sqrt(np.linalg.det(cov)))
    #计算PDF的指数部分
    exponet = -0.5 * np.dot(np.dot((x-mean).T, np.linalg.inv(cov)),(x - mean))
    return coeff * np.exp(exponet)


# 1.散点输入
class1_points = np.array([[1.9, 1.2],
                          [1.5, 2.1],
                          [1.9, 0.5],
                          [1.5, 0.9],
                          [0.9, 1.2],
                          [1.1, 1.7],
                          [1.4, 1.1]])

class2_points = np.array([[3.2, 3.2],
                          [3.7, 2.9],
                          [3.2, 2.6],
                          [1.7, 3.3],
                          [3.4, 2.6],
                          [4.1, 2.3],
                          [3.0, 2.9]])
#合并数据集,创造标签
X = np.concatenate((class1_points,class2_points))
y = np.concatenate((np.zeros(len(class1_points)),np.ones(len(class2_points))))

#2.计算先验概率(每一类别的数据在数据集中的比例)
prior_probabilities = [np.sum(y == 0) / len(y),np.sum(y == 1)/len(y)]
# print(prior_probabilities)

#3.计算高斯分布的概率密度函数
#求解均值
# print(X)
class_means = [np.mean(X[y==0],axis=0),np.mean(X[y==1],axis=0)]
# print(class_mean)

X_y_0 = X[y==0].T
X_y_1 = X[y==1].T
class_covs = [np.cov(X_y_0),np.cov(X_y_1)]

point = np.array([0.5,3])


xx,yy = np.meshgrid(np.arange(0,5,0.05),np.arange(0,4,0.05))

#预测网格点
grid_points = np.c_[xx.ravel(),yy.ravel()]
grid_label = []
for point in grid_points:
    posterior_probabilities = []
    for i in range(2):
        #使用高斯分布的概率密度函数求解条件概率
        likelihood = pdf(point,class_means[i],class_covs[i])
        #4.得到后验概率,比较大小,获得分类
        posterior_probabilities.append(prior_probabilities[i] * likelihood)

    #比较大小,获得分类
    pre_class = np.argmax(posterior_probabilities)
    grid_label.append(pre_class)
    #5.显示决策边界
    #预测的标签与xx他们形状一致

grid_label = np.array(grid_label).reshape(xx.shape)

plt.scatter(class1_points[:,0],class1_points[:,1],c='blue',label="class 1")
plt.scatter(class2_points[:,0],class2_points[:,1],c="red",label="class 2")
# plt.scatter(point[0],point[1],c="green",label="point")

contour = plt.contour(xx,yy,grid_label,levels=[0.5],colors="green")

plt.legend()
# if pre_class == 0:
#     plt.text(point[0] + 0.1,point[1] - 0.1,"class 1")
# else:
#     plt.text(point[0] + 0.1,point[1] - 0.1,"class 2")
plt.show()