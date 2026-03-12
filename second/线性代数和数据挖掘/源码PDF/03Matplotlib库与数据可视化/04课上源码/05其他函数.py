# import numpy as np
# import matplotlib.pyplot as plt
#
# # 计算曲线上的点的x和y坐标
# x = np.arange(0, 3 * np.pi, 0.1)
# y = np.sin(x)
#
# # 使用Matplotlib绘制点，并添加fmt和kwargs属性
# plt.plot(x, y,
#          label='Sine Wave',  # 图例标签
#          linewidth=2,  # 线宽
#          linestyle='-',  # 线型
#          color='blue',  # 线的颜色
#          marker='o',  # 标记样式
#          markersize=5,  # 标记的大小
#          markeredgecolor='black',  # 标记边缘的颜色
#          markeredgewidth=1,  # 标记边缘的宽度
#          markerfacecolor='none',  # 标记内部的颜色
#          alpha=0.5 # 透明度
#          )
#
# # 显示x轴标签
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sine Wave')
#
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.arange(0, 3 * np.pi, 0.01)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 在第一个位置创建子图
plt.subplot(2, 2, 1)  # 2行1列，第一个子图
plt.plot(x, y_sin)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('y_sin')

# 在第二个位置创建子图
plt.subplot(2, 2, 2)  # 2行1列，第二个子图
plt.plot(x, y_cos)
plt.title('Cosine Wave')
plt.xlabel('x')
plt.ylabel('y_cos')

# 调用 tight_layout 来自动调整子图参数
plt.tight_layout()

# 显示图形
plt.show()