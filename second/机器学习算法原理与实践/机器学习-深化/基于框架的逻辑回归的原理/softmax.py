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
        self.fc = torch.nn.Linear(2,2)

    def forward(self,x):
        x = self.fc(x)
        #dim 指定了softmax在哪个维度上进行
        #dim = 1 表示在第二个维度上进行,列
        #对于每一行的元素,他保持每一列的元素和为1
        #example:x (sample,num_classes),
        #toch.softmax(x,dim=1),pytoch在每个样本上的类别进行softmax,保证每个样本的所有类别的概率和为1
        return torch.softmax(x,dim=1)

model = LogisticRegreModel()

#3.定义损失函数,优化器
criterion = torch.nn.CrossEntropyLoss()#交叉熵损失
optimizer = torch.optim.SGD(model.parameters(),lr=0.05)#随机梯度下降

plt.figure(figsize=(12,6))
xx,yy = np.meshgrid(np.arange(0,6,0.1),np.arange(0,6,0.1))
grid_points = np.c_[xx.ravel(),yy.ravel()]

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
# epoch_list = []
# loss_list = []
#4.开始迭代
epoches = 1000
for epoch in range(1,epoches + 1):
    inputs = torch.tensor(x_train,dtype=torch.float32)
    labels = torch.tensor(Y_train,dtype=torch.long)

    outputs = model(inputs)
    loss = criterion(outputs,labels)
    #反向传播与优化
    optimizer.zero_grad()#清空梯度
    loss.backward()#反向传播计算梯度
    optimizer.step()#更新参数
    # 5.显示频率设置
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch:{epoch},loss:{loss}")
        grid_tensor =torch.tensor(grid_points,dtype=torch.float32)
        Z = model(grid_tensor).detach().numpy()
        Z = Z[:,1]
        Z = Z.reshape(xx.shape)

        plt.cla()
        plt.scatter(class1_points[:,0],class1_points[:,1])
        plt.scatter(class2_points[:,1],class2_points[:,1])
        plt.contour(xx,yy,Z,levels=[0.5])
        plt.pause(1)
        plt.show()


        # print(model.fc.weight.data)


#6.绘制

