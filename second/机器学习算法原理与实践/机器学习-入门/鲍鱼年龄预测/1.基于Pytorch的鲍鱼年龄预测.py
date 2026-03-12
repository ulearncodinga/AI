import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#1.加载数据集
data = pd.read_csv('./abalone.data')
print(data.head())

#加入列名称
column_names = ['sex','Length','Diameter','Height','Whole_weight','Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
data.columns = column_names
#因为sex有F,M,I类别,是分类变量
data = pd.get_dummies(data,columns=['sex'])
print(data.keys())
X = data[['sex_F', 'sex_M', 'sex_I', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]
y = data['Rings']#把Ring作为目标变量

#分为训练集和测试集
# random_state为随机数种子,保证每次划分结果已知,用来复现和对比不同实验情况
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

#将数据类型转化为Pytorch张量

X_train_tensor = torch.tensor(X_train_scaler,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test_scaler,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32).view(-1,1)



#定义回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self,input_size):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_size,1)

    def forward(self,x):
        return self.linear(x)

#实例化模型
input_size = X_train_tensor.shape[1]
model = LinearRegressionModel(input_size)
#定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.1)

#训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs,y_train_tensor)

    #反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f'Epoch[{epoch + 1}/{num_epochs}],Loss:{loss.item():.4f}')



#评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions,y_test_tensor)



# 再把Pytorch张量形式转换为Numpy数组形式,方便之后用matplotlib绘图
predictions = predictions.numpy()
y_test_numpy = y_test_tensor.numpy()


#绘图
plt.scatter(y_test_numpy,predictions,color='r')
plt.plot([min(y_test_numpy),max(y_test_numpy)],[min(y_test_numpy),max(y_test_numpy)],linestyle='--',color='b',linewidth=3)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title('Regression Results')
plt.show()