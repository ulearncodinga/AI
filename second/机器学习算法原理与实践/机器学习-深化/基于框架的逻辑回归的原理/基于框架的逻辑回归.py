''''''
'''散点输入->定义前向模型->定义损失函数和优化器->开始迭代->
显示频率设置->决策边界绘制与输出
'''
import numpy as np
import matplotlib.pyplot as plt
import torch

#1.散点输入
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

x_train = np.concatenate((class1_points,class2_points))
Y_train = np.concatenate((np.zeros(len(class1_points)),np.ones(len(class2_points))))


#2.定义前向模型
torch.manual_seed(1)#随机数种子

class LogisticRegreModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegreModel,self).__init__()
        self.fc = torch.nn.Linear(2,1)

    def forward(self,x):
        x = self.fc(x)
        return torch.sigmoid(x)

model = LogisticRegreModel()

#3.定义损失函数,优化器
criterion = torch.nn.BCELoss()#二分类交叉熵损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)#随机梯度下降


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
epoch_list = []
loss_list = []
#4.开始迭代
epoches = 1000
for epoch in range(1,epoches + 1):
    inputs = torch.tensor(x_train,dtype=torch.float32)
    labels = torch.tensor(Y_train,dtype=torch.float32).view(-1,1)

    outputs = model(inputs)
    loss = criterion(outputs,labels)
    #反向传播与优化
    optimizer.zero_grad()#清空梯度
    loss.backward()#反向传播计算梯度
    optimizer.step()#更新参数
    # 5.显示频率设置
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch:{epoch},loss:{loss}")
        w1,w2 = model.fc.weight.data.flatten()
        b = model.fc.bias.data[0]
        #斜率
        slope = -w1/w2
        #截距
        intercept = -b / w2
        #绘制直线
        x_min,x_max = 0,5
        x = np.array([x_min,x_max])
        y = slope * x + intercept

        #绘制
        ax1.clear()
        ax1.scatter(class1_points[:,0],class1_points[:,1])
        ax1.scatter(class2_points[:,0],class2_points[:,1])
        ax1.plot(x,y)

        ax2.clear()
        epoch_list.append(epoch)
        loss_list.append(loss.item())
        ax2.plot(epoch_list,loss_list)
        plt.pause(1)
plt.show()



#6.绘制

