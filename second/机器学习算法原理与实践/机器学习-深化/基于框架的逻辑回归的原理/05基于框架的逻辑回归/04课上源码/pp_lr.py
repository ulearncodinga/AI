import numpy as np
import paddle
import matplotlib.pyplot as plt

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

X_train = np.concatenate((class1_points, class2_points))
Y_train = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points))))

# 2.定义前向模型
class LogisticRegreModel(paddle.nn.Layer):
    def __init__(self):
        super(LogisticRegreModel, self).__init__()
        self.fc = paddle.nn.Linear(2, 1)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)

model = LogisticRegreModel()

# 3.定义损失函数 和优化器
cri = paddle.nn.BCELoss()
optimizer = paddle.optimizer.SGD(0.05, model.parameters())

fig, (ax1, ax2) = plt.subplots(1, 2 , figsize=(12, 6))
epoch_list = []
loss_list = []
# 4.开始迭代
epoches = 1000
for epoch in range(1, epoches + 1):
    inputs = paddle.to_tensor(X_train, dtype=paddle.float32)
    labels = paddle.to_tensor(Y_train, dtype=paddle.float32)

    outputs = model(inputs)
    loss = cri(outputs, labels.unsqueeze(1))
    # 反向传播与优化
    optimizer.clear_grad()
    loss.backward()
    optimizer.step()

    # 5.显示频率设置
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch: {epoch}, loss: {loss}")
        w1, w2 = model.fc.weight.numpy()
        b = model.fc.bias.numpy()[0]

        # 斜率和截距求解
        slope = -w1 / w2
        intercept = -b / w2
        # 绘制直线
        x_min, x_max = 0, 5
        x = np.array([x_min, x_max])
        y = slope * x + intercept

        # 6.绘制
        ax1.clear()
        ax1.scatter(class1_points[:, 0], class1_points[:, 1])
        ax1.scatter(class2_points[:, 0], class2_points[:, 1])
        ax1.plot(x, y)

        ax2.clear()
        epoch_list.append(epoch)
        loss_list.append(loss.detach().numpy())
        ax2.plot(epoch_list, loss_list)
        plt.pause(1)
plt.show()



















