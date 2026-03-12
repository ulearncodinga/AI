# 导入相关库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# 1.散点输入
points = np.array([[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6],
     [0.4, 34.0], [0.8, 62.3]])

# 分离特征和标签
X = points[:, 0]
Y = points[:, 1]

# 2.参数初始化
w = 0
b = -1
lr = 0.001

# 3.定义损失函数
def loss_func(X, w, b):
    pre_y = np.dot(X, w) + b
    total_loss = np.mean((pre_y - Y) ** 2)
    return total_loss


# 4.定义优化器
def SGD(points, w, b, lr, batch_size):
    # 将数据的顺序打乱
    np.random.shuffle(points)
    for num_batch in range(0, len(points), batch_size):
        batch_points = points[num_batch:num_batch+batch_size, :]
        batch_x = batch_points[:, 0]
        batch_y = batch_points[:, 1]

        # 计算梯度
        batch_pre_y = w * batch_x + b
        dw = np.mean(2 * (batch_pre_y - batch_y) * batch_x)
        db = np.mean(2 * (batch_pre_y - batch_y))

        w -= lr * dw
        b -= lr * db
    return w, b


# 构建网格点
w_values = np.linspace(-20, 80, 100)
b_values = np.linspace(-20, 80, 100)
W, B = np.meshgrid(w_values, b_values)
loss_values = np.zeros_like(W)

# 计算网格中每个点的损失值
for i, w in enumerate(w_values):
    for j, b in enumerate(b_values):
        loss_values[j, i] = loss_func(X, w, b)

# 构建图像对象
fig = plt.figure(figsize=(12, 6))
# 创建画布2*2
gs = gridspec.GridSpec(2, 2)

# 左上角格子
ax1 = fig.add_subplot(gs[0, 0])

# 左下角格子
ax2 = fig.add_subplot(gs[1, 0])

# 整个右侧格子
ax3 = fig.add_subplot(gs[:, 1], projection="3d")
ax3.plot_surface(W, B, loss_values, cmap="viridis", alpha=0.8)

# 存储梯度下降路径
gd_path = []

# 5.开始迭代
epoches = 1000
bs = 10
for epoch in range(1, epoches+1):
    gd_path.append((w, b))
    w, b = SGD(points, w, b, lr, batch_size=bs)
    if epoch == 1 or epoch % 20 == 0:
        # 6.设置显示频率
        print(loss_func(X, w, b))
        # 7.绘图与显示
        ax1.clear()
        ax1.scatter(X, Y, c="r")
        x_line = np.linspace(np.min(X), np.max(X), 2)
        y_line = np.dot(x_line, w) + b
        ax1.plot(x_line, y_line, c="g")

        ax2.clear()
        ax2.contourf(W, B, loss_values, levels=50, camp="viridis")
        ax2.scatter(w, b, c="black", s=20)
        gd_w, gd_b = zip(*gd_path)
        ax2.plot(gd_w, gd_b, c="black")

        ax3.scatter(w, b, loss_func(X, w, b), c="black", s=20)
        ax3.plot(gd_w, gd_b, [loss_func(X, gd_w[i], gd_b[i]) for i in range(len(gd_w))], c="black")

        plt.pause(1)
plt.show()










