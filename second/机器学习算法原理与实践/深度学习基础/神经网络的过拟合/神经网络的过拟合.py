''''''
'''
训练数据集上表现得很好,测试数据上表现得很差

造成过拟合的原因:
数据量不足
模型过于复杂
正则化强度不足
散点输入->定义前向模型->定义损失函数和优化器->开始迭代->显示频率设置->拟合线显示和输出
'''
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 1.创造数据，数据集
points = np.array([[-0.5, 7.7], [1.2, 65.8], [0.4, 39.2], [-1.4, -15.7], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]])
#分离特征与标签
x_train = points[:,0]
y_train = points[:,1]

#2.定义前向模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.layer1 = nn.Linear(1,16)
        self.layer2 = nn.Linear(16,32)
        self.layer3 = nn.Linear(32,16)
        self.layer4 = nn.Linear(16,1)

    def forward(self,x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x
model = Model()


#3.定义损失函数和优化器
lr =0.05
cri = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)


#4.开始迭代
epoches = 1000
for epoch in range(1,epoches + 1):
    x_train_tensor = torch.tensor(x_train,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train,dtype=torch.float32)
    #前向传播
    y_pred = model(x_train_tensor.unsqueeze(1))
    #计算损失
    loss= cri(y_pred.squeeze(1),y_train_tensor)
    #反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch == 1 or epoch % 20 == 0:
        #绘图
        plt.cla()
        plt.scatter(x_train,y_train)
        x_range = torch.tensor(np.linspace(-2,2,300),dtype=torch.float32).unsqueeze(1)
        y_range = model(x_range).detach().numpy()
        plt.plot(x_range,y_range)
        print(f"epoch:{epoch},Loss:{loss}")
        plt.pause(0.2)

plt.show()
#5.显示频率设置

#.拟合线显示和输出
