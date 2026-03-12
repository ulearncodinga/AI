# 导入必要的库（保持不变）
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init


# -------------------------- 核心：生成月牙分布数据 --------------------------
def generate_moon_data(n_samples=200, noise=0.1, random_state=42):
    """
    生成月牙分布数据（类似sklearn的make_moons，但手动实现更灵活）
    :param n_samples: 每类样本数
    :param noise: 噪声强度（越大越分散）
    :param random_state: 随机种子（保证可复现）
    :return: class1_points（上月牙）, class2_points（下月牙）
    """
    np.random.seed(random_state)
    # 极坐标参数：固定半径范围，角度分两部分（上月牙+下月牙）
    radius = np.random.uniform(1.5, 3.5, n_samples)  # 半径范围[1.5, 3.5]
    angle1 = np.random.uniform(0, np.pi, n_samples)  # 上月牙角度：0~π（180°）
    angle2 = np.random.uniform(np.pi, 2 * np.pi, n_samples)  # 下月牙角度：π~2π（180°~360°）

    # 极坐标转直角坐标（x, y），添加噪声
    # 上月牙（class1）：圆心偏移(4, 4)，让月牙居中在画布中间
    class1_x = 4 + radius * np.cos(angle1) + np.random.normal(0, noise, n_samples)
    class1_y = 4 + radius * np.sin(angle1) + np.random.normal(0, noise, n_samples)
    class1_points = np.column_stack((class1_x, class1_y))

    # 下月牙（class2）：同样偏移(4, 4)，与上月牙交错
    class2_x = 4 + radius * np.cos(angle2) + np.random.normal(0, noise, n_samples)
    class2_y = 4 + radius * np.sin(angle2) + np.random.normal(0, noise, n_samples)
    class2_points = np.column_stack((class2_x, class2_y))

    return class1_points, class2_points


# 生成月牙数据（每类200个样本，噪声0.15，保证分布清晰且有一定难度）
class1_points, class2_points = generate_moon_data(n_samples=200, noise=0.15, random_state=42)

# 验证数据形状（可选）
print("上月牙样本数：", len(class1_points))
print("下月牙样本数：", len(class2_points))
print("数据形状示例：", class1_points[:3])  # 输出前3个样本的(x,y)坐标
# （接上面的数据生成代码）
# -------------------------- 数据划分（保持规范，按10%划分测试集） --------------------------
np.random.shuffle(class1_points)
np.random.shuffle(class2_points)
split_idx1 = int(0.1 * len(class1_points))
split_idx2 = int(0.1 * len(class2_points))

# 训练集（90%）+ 测试集（10%）
class1_train = class1_points[split_idx1:]
class2_train = class2_points[split_idx2:]
class1_test = class1_points[:split_idx1]
class2_test = class2_points[:split_idx2]

# 合并特征和标签
train_points = np.concatenate((class1_train, class2_train), axis=0)
test_points = np.concatenate((class1_test, class2_test), axis=0)
train_labels = np.concatenate((np.zeros(len(class1_train)), np.ones(len(class2_train))), axis=0)
test_labels = np.concatenate((np.zeros(len(class1_test)), np.ones(len(class2_test))), axis=0)

# 转换为Tensor
train_points_tensor = torch.tensor(train_points, dtype=torch.float32)
test_points_tensor = torch.tensor(test_points, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# -------------------------- 模型适配：增强非线性拟合能力 --------------------------
class MoonClassificationModel(nn.Module):
    def __init__(self):
        super(MoonClassificationModel, self).__init__()
        # 增加隐藏层神经元数量（2→16→8→2），增强非线性表达
        self.layer1 = nn.Linear(2, 16)  # 输入层→隐藏层1（16个神经元）
        self.layer2 = nn.Linear(16, 8)  # 隐藏层1→隐藏层2（8个神经元）
        self.layer3 = nn.Linear(8, 2)   # 隐藏层2→输出层（2类分类）

        # 权重初始化（适配ReLU）
        init.kaiming_uniform_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.layer1.bias)
        init.kaiming_uniform_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.layer2.bias)
        init.kaiming_uniform_(self.layer3.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.layer3.bias)

    def forward(self, x):
        # 用ReLU激活函数，缓解梯度消失
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  # 输出层：直接输出logits（CrossEntropyLoss自动处理softmax）
        return x

# 实例化模型、损失函数、优化器
model = MoonClassificationModel()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适配多分类）
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # 降低学习率，减轻过拟合

# -------------------------- 可视化初始化 --------------------------
x_min, x_max = 0, 8  # 适配月牙中心(4,4)，扩大范围方便观察
y_min, y_max = 0, 8
step_size = 0.05  # 加密网格，让决策边界更平滑
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                     np.arange(y_min, y_max, step_size))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

# 创建图形（1行2列：决策边界+损失曲线）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
step_list = []
loss_list = []
test_step_list = []
test_loss_list = []

# -------------------------- 训练迭代（适配月牙分类，增加轮数到1500） --------------------------
num_iterations = 1500
for n in range(num_iterations + 1):
    # 训练阶段
    model.train()
    optimizer.zero_grad()
    outputs = model(train_points_tensor)
    loss = criterion(outputs, train_labels_tensor)
    loss.backward()
    optimizer.step()

    # 记录训练损失
    step_list.append(n)
    loss_list.append(loss.item())

    # -------------------------- 每50轮更新可视化 --------------------------
    frequency_display = 50
    if n % frequency_display == 0 or n == 1:
        # 1. 绘制月牙决策边界（左图）
        model.eval()
        with torch.no_grad():
            Z = model(grid_tensor).detach().numpy()
            Z = Z[:, 1]  # 取类别1（下月牙）的概率
            Z = Z.reshape(xx.shape)

        ax1.clear()
        # 训练集：上月牙（蓝色）、下月牙（红色）
        ax1.scatter(class1_train[:, 0], class1_train[:, 1], c='blue', s=20, label='Upper Moon (Train, Class 0)')
        ax1.scatter(class2_train[:, 0], class2_train[:, 1], c='red', s=20, label='Lower Moon (Train, Class 1)')
        # 测试集：星形标记，区分训练集
        ax1.scatter(class1_test[:, 0], class1_test[:, 1], c='lightblue', s=60, marker='*', label='Upper Moon (Test, Class 0)')
        ax1.scatter(class2_test[:, 0], class2_test[:, 1], c='lightcoral', s=60, marker='*', label='Lower Moon (Test, Class 1)')
        # 决策边界（概率=0.5，红色实线）
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, label='Decision Boundary')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Moon Classification (Iteration {n})')
        ax1.legend()
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 8)
        ax1.grid(True, alpha=0.3)

        # 2. 绘制损失曲线（右图）
        with torch.no_grad():
            test_outputs = model(test_points_tensor)
            test_loss = criterion(test_outputs, test_labels_tensor)
        test_step_list.append(n)
        test_loss_list.append(test_loss.item())

        ax2.clear()
        ax2.plot(step_list, loss_list, 'r-', label='Train Loss')
        ax2.plot(test_step_list, test_loss_list, 'b-', label='Test Loss')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Train vs Test Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.pause(0.1)

plt.tight_layout()
plt.show()

# 最终测试集准确率（可选）
model.eval()
with torch.no_grad():
    test_preds = torch.argmax(model(test_points_tensor), dim=1)
    accuracy = (test_preds == test_labels_tensor).float().mean().item()
print(f"最终测试集准确率：{accuracy:.4f}")