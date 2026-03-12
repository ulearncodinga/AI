# 导入相关库
import numpy as np
import matplotlib.pyplot as plt

# 1.散点输入
points = np.array([[0.8, 0], [1.1, 0], [1.7, 0], [1.9, 0], [2.7, 1], [3.2, 1], [3.7, 1], [4.0, 1], [5.0, 0], [5.5, 0], [6.0, 0], [6.3, 0]])

# 将输入样本中的特征和标签分开
X = points[:, 0]
y = points[:, 1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2.前向计算
def forward(w11_1, b1_1, w12_1, b2_1, w11_2, w21_2, b1_2, X):
    z1_1 = w11_1 * X + b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w12_1 * X + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_1, a2_1, a1_2


# 3.参数初始化
w11_1, b1_1, w12_1, b2_1, w11_2, w21_2, b1_2 = 0.1, 0.6, 0.9, 0, -1.5, 0.1, 0.9
lr = 0.5

# 4.损失函数
def loss_func(y, y_hat):
    loss = np.mean((y - y_hat) ** 2)
    return loss

x_values = np.linspace(0, 7, 100)
loss_list = []
# 5.开始迭代
epoches = 5000
for epoch in range(1, epoches + 1):
    # 计算损失函数
    a1_1, a2_1, a1_2 = forward(w11_1, b1_1, w12_1, b2_1, w11_2, w21_2, b1_2, X)
    loss = loss_func(y, a1_2)
    loss_list.append(loss)

    # 6.反向传播
    deda1_2 = -2 * (y - a1_2)
    dedz1_2 = deda1_2 * a1_2 * (1 - a1_2)

    dedw11_2 = np.mean(dedz1_2 * a1_1)
    dedw21_2 = np.mean(dedz1_2 * a2_1)
    dedb1_2 = np.mean(dedz1_2)

    deda1_1 = dedz1_2 * w11_2
    dedz1_1 = deda1_1 * a1_1 * (1 - a1_1)
    dedw11_1 = np.mean(dedz1_1 * X)
    dedb1_1 = np.mean(dedz1_1)

    deda2_1 = dedz1_2 * w21_2
    dedz2_1 = deda2_1 * a2_1 * (1 - a2_1)
    dedw12_1 = np.mean(dedz2_1 * X)
    dedb2_1 = np.mean(dedz2_1)

    # 梯度下降获得更新
    w11_2 -= lr * dedw11_2
    w21_2 -= lr * dedw21_2
    b1_2 -= lr * dedb1_2

    w11_1 -= lr * dedw11_1
    b1_1 -= lr * dedb1_1
    w12_1 -= lr * dedw12_1
    b2_1 -= lr * dedb2_1
    # 7.显示频率设备
    if epoch == 1 or epoch % 100 == 0:
        print(f"epoch: {epoch}, loss: {loss}")
        _, _, y_values = forward(w11_1, b1_1, w12_1, b2_1, w11_2, w21_2, b1_2, x_values)
        # 8.梯度下降显示
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(x_values, y_values)
        plt.scatter(X, y)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(loss_list)), loss_list)
        plt.pause(1)
plt.show()


















