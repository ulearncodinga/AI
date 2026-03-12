# import matplotlib.pyplot as plt
#
#
#
# plt.plot([0, 1], [1, 2])
# fig = plt.figure(num=1, clear=True)
# plt.plot([0, 1], [2, 3])
#
# plt.show()
#


import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.arange(-10, 10, 0.01)
y1 = np.sin(x) + 100
y2 = np.cos(x)
y3 = np.tan(x)
y4 = 1 / (1 + np.exp(-x))  # Sigmoid function

# 创建一个 2x2 的子图网格
fig, axs = plt.subplots(2, 2)


# # 第一个子图
# axs[0, 0].plot(x, y1)
# axs[0, 0].set_title('Sine Wave')
# axs[0, 0].set_xlabel('X-axis')
# axs[0, 0].set_ylabel('Y-axis')
#
# # 第二个子图
# axs[0, 1].plot(x, y2)
# axs[0, 1].set_title('Cosine Wave')
# axs[0, 1].set_xlabel('X-axis')
# axs[0, 1].set_ylabel('Y-axis')
#
# # 第三个子图
# axs[1, 0].plot(x, y3)
# axs[1, 0].set_title('Tangent Wave')
# axs[1, 0].set_xlabel('X-axis')
# axs[1, 0].set_ylabel('Y-axis')
#
# # 第四个子图
# axs[1, 1].plot(x, y4)
# axs[1, 1].set_title('Sigmoid Function')
# axs[1, 1].set_xlabel('X-axis')
# axs[1, 1].set_ylabel('Y-axis')

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()

