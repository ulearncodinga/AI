import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 1. 生成二维网格数据（关键修正）
xs = np.linspace(-10, 10, 100)  # x轴数据，生成100个点
ys = np.linspace(-10, 10, 100)  # y轴数据，生成100个点
X, Y = np.meshgrid(xs, ys)  # 转换为二维网格
a = X.reshape(10000)
b = Y.reshape(10000)

# 2. 计算对应的Z值（每个网格点的高度）
Z = np.sqrt(X ** 2 + Y ** 2)  # 这里是圆锥面的公式

# 3. 绘制等高线图
# 可选：plt.contourf() 绘制填充的等高线
contour = plt.contour(X, Y, Z, levels=15)  # levels控制等高线数量

# 添加颜色条和标签
# plt.colorbar(contour, label='Z值')
plt.xlabel('X轴')
plt.ylabel('Y轴')
# plt.zlabel('Z轴')
# plt.title('等高线图示例 (Z = √(X² + Y²))')
print(X.shape)
print(Y.shape)
print(Z.shape)
plt.show()
