import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random
import os
import torch.nn as nn
import torch.optim as optim
import torch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader


#设置随机种子保证结果的可重复性
def setup_seed(seed):
    #设置numpy随机数种子,确保numpy生成的随机数序列一致
    np.random.seed(seed)

    #设置python内的随机数种子,保证Python内置的随机函数生成的随机数一致
    random.seed(seed)

    #设置Python哈希种子,避免不同运行环境夏哈希结果不同,影响随机数生成
    os.environ['PYTHONHASHSEED'] = str(seed)

    #设置Pytorch随机种子,使Pytorch生辰大哥随机数序列可以重复
    torch.manual_seed(seed)

    #检查是否有可用的CUDA设备(GPU)
    if torch.cuda.is_available():
        #设置CUDA随机种子,保证在GPU上随即操作可以重复
        torch.cuda.manual_seed(seed)
        #为所有GPU设置随机种子
        torch.cuda.manual_seed_all(seed)
        #关闭cudnn自动寻找最优算法加速的功能,保证结果可重复
        torch.backends.cudnn.benchmark = False
        #设置cudnn为确定性算法,确保每次运行结果一致
        torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device ='cuda'
    print('CUDA is useful')
else:
    device = 'cpu'
    print('CUDA is FALSE')

setup_seed(0)



#设置pandas显示选项,以便于显示更多的列和行的内容
#最多选择1000列
pd.set_option('display.max_columns',1000)
#显示宽度为1000
# pd.set_option('display_width',1000)
#每列最多显示1000个字符
pd.set_option('display.max_colwidth',1000)

#读取数据集,需要注意的是,编码格式必须为big5
train_data = pd.read_csv('./dataset/train.csv',encoding='big5')
#查看前五行数据
# print(train_data.head())
#打印数据集的信息,查看数据集的情况
# print(train_data.info())#打印出来看到没有缺失值

#选取从第三列开始到最后的所有列作为特征数据
train_data = train_data.iloc[:,3:]
#将数组中值为NR的元素替换为0
train_data[train_data == 'NR'] = 0


#将train_data转换为numpy数组
numpy_data = train_data.to_numpy()


#检查数据集中缺失值的情况
# print(train_data.isnull().sum())


#创建一个链表,用来存储拆分后的数据
datas = []
#按照 步长为18 分割数据
for i in range(0,4320,18):
    datas.append(numpy_data[i:i+18,:])
# print(datas)
#将datas转换为Numpy数组
datas_array = np.array(datas,dtype=float)

#对数据进行维度变换和重塑,转换为DataFrame数组
test =pd.DataFrame(datas_array.transpose(1,0,2).reshape(18,-1).T)
# print(test.shape)

#计算特征相关性矩阵(找出特征相关性强的特征)
corr = train_data.corr()

#绘制相关热性图
plt.figure(0)
#seaborn库:基于matplotlib的高级可视化库
#heatmap:热力图,用于展示矩阵数据,通过颜色的深浅表示数值大小,常用于展示相关性矩阵,混淆矩阵等内容
#帮助用户更直观的发现数据之间的关系
sns.heatmap(corr,annot=True)

#从相关性矩阵中筛选比较重要的特征
important_feature = []
for i in range(len(corr.columns)):
    #选择与第九列相关性系数绝对值大于0.2的特征
    if abs(corr.iloc[i,9])>0.2:
        important_feature.append(corr.columns[i])
print('比较重要的特征:',important_feature)


#确定目标特征所在列后,就要去将数据集划分特征和目标了
#选取重要特征,但是要排除掉目标本身,剩下的所有的重要特征就是我们的目标特征
X = train_data[important_feature].drop(9,axis=1)
#选取第9列作为目标
y = train_data[9]


#划分训练集和测试集
train_radio = 0.8
#训练集特征
X_train = X[:int(train_radio * len(train_data))]
#测试集特征
X_test = X[int(train_radio * len(train_data)):]
#训练集标签
y_train = y[:int(train_radio * len(train_data))]
#测试机特征
y_test = y[int(train_radio * len(train_data)):]

#使用标准化进行特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#将数据转换为Pytorch的张量
#训练集的特征张量表示
X_train_tensor = torch.tensor(X_train_scaled,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)
#测试集的特征张量表示
X_test_tensor = torch.tensor(X_test_scaled,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32).view(-1,1)


#使用DataLoader去加载数据集
#创建TensorDataset对象,将训练集转换为特征张量与标签,组合在一起
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)

#创建DataLoader 用于批量处理加载数据
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)


#构建DNN模型
class DNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        #调用父类构造函数
        super(DNN,self).__init__()
        #定义第一个Linear层,输入就是input_size,输出维度就是hidden_size
        self.fc1 = nn.Linear(input_size,hidden_size)
        #定义第二个Linear层,输入是hidden_size,输出是hidden_size
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        #定义第三个Linear层,输入时hidden_size输出是hidden_size
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        #定义第四个Linear层,输入时hidden_size输出是output_size
        self.fc4 = nn.Linear(hidden_size,output_size)

        #定义激活函数
        self.relu = nn.ReLU()
    def forward(self,x):
        #输入数据经过第一个Linear和ReLu激活
        x = self.relu(self.fc1(x))
        #输入数据经过第二个Linear和ReLu激活
        x = self.relu(self.fc2(x))
        #输入数据经过第三个Linear和ReLu激活
        x = self.relu(self.fc3(x))
        #输入数据经过第四个Linear和ReLu激活
        x = self.relu(self.fc4(x))
        return x

#实例化模型
#需要获取input_size,hidden_size,outputs_size
#将输入特征的列数作为输入的维度
input_size = X_train_tensor.shape[1]
#hidden_size自由发挥
hidden_size = 256
#output_size 标签的维度
output_size = 1


#模型实例化成功
model = DNN(input_size,hidden_size,output_size)
#将模型放到device上
model.to(device)

#定义损失函数
criterion = nn.MSELoss()
#定义优化器
optimizer = optim.Adam(model.parameters(),lr=0.01)

#定义训练的epoch
num_epochs = 1000

for epoch in range(num_epochs):
    #将模型设置为训练模型
    model.train()

    #初始化总损失
    total_loss = 0

    #循环加载DataLoader,每次获取一个数据和标签
    for inputs,labels in train_loader:
        #清理优化器梯度
        optimizer.zero_grad()

        #将inputs和labels 放到device上
        inputs = inputs.to(device)
        labels = labels.to(device)
        #前向传播,计算输出
        outputs = model(inputs)

        #计算损失
        loss = criterion(outputs,labels)
        #反向传播和优化
        loss.backward()
        #更新参数
        optimizer.step()
        #累加当前批次的损失
        total_loss += loss
    #计算本次epoch的平均损失
    avg_loss = total_loss / len(train_loader)

    #每十个epoch打印一次当前模型的训练情况
    if(epoch + 1)%10 == 0:
        print(f'epoch:{epoch},loss:{avg_loss:.4f}')



plt.show()