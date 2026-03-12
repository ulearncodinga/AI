import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset,DataLoader
#1.读取数据
dataset = pd.read_csv('./dataset/spambase.data',header=None)
#查看读取进来的数据前几行,确保数据集加载正确
# print(dataset.head())




#2.数据预处理
#提取特征数据
X = dataset.iloc[:,:-1]
#提取标签数据
y = dataset.iloc[:,-1]

#划分训练集和测试机
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#创建标准化对象,对原始数据进行数据标准化的操作
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#将标准化后的训练集与测试集转化为tensor
X_train_tensor = torch.tensor(X_train_scaled,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)

#DataLoader:可以让我们搞笑的从数据集中批量加载数据 batch
#使用之前,先将训练集特征张量与标签张量组合在一起 TensorDataset类
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)

#创建DataLoader,用于批量加载数据,我们可以自定义每个批次加载多少样本,batch_size,shuffle
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

#3.模型训练

#4.模型评估

#5.数据可视化