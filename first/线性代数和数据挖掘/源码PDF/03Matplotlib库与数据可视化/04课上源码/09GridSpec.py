import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(8, 4))

# 创建一个 2 行 3 列的 GridSpec
gs = GridSpec(2, 3)


# 添加子图
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, :])
# 注意：gs[1, 1:] 表示第二行的第二列和第三列

plt.show()