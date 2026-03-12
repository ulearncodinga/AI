import mplcursors
import numpy as np
from matplotlib import pyplot as plt

# 注释或删除Jupyter魔法命令
# %matplotlib notebook

# 关闭所有现有的图表
plt.close('all')


# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


xs = np.linspace(-10, 10, 100)
ys = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(xs, ys)  # 生成网格点坐标矩阵
# 计算Z轴的值 (X^2+Y^2)^1/2 = Z
Z = np.sqrt(X**2 + Y**2)

# 创建画布和3D轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 创建1行1列的子图，选择第一个子图

# 绘制3D图
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(surf, label='Z 值')

# 显示图形
plt.show()
