# import numpy as np
# import matplotlib.pyplot as plt
#
# # 计算曲线上的点的x和y坐标
# # x = np.arange(0, 3 * np.pi, 0.1)
# x = np.linspace(0, 3 * np.pi, 1000)
# y = np.cos(x)
#
# # 使用Matplotlib绘制点，并添加fmt和kwargs属性
# plt.plot(x, y, '-',
#          label='Cosine Wave',  # 图例标签
#          linewidth=2,  # 线宽
#          color='blue',  # 线的颜色
#          marker='o',  # 标记样式
#          markersize=5,  # 标记的大小
#          markeredgecolor='black',  # 标记边缘的颜色
#          markeredgewidth=1,  # 标记边缘的宽度
#          markerfacecolor='none',  # 标记内部的颜色
#          alpha=0.5 # 透明度
#          )
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()






# import numpy as np
# import matplotlib.pyplot as plt
#
# # 只有两个点的x和y坐标
# x = np.array([0, 5, 6, 7, 8])  # 两个点的x坐标
# y = np.array([0, 10, 7, 12, 1])        # 对应的y坐标
#
# # 使用Matplotlib绘制两个点的直线，并添加kwargs属性
# plt.plot(x, y, '-',
#          label='Line between two points',  # 图例标签
#          linewidth=2,  # 线宽
#          color='red',  # 线的颜色
#          marker='o',  # 标记样式
#          markersize=10,  # 标记的大小
#          markeredgecolor='black',  # 标记边缘的颜色
#          markeredgewidth=2,  # 标记边缘的宽度
#          markerfacecolor='blue',  # 标记内部的颜色
#          alpha=1.0  # 透明度
#          )
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()









# import numpy as np
# import matplotlib.pyplot as plt
#
#
#
# plt.plot([1, 2, 3, 4, 5], marker='o')
#
# plt.show()









import numpy as np
import matplotlib.pyplot as plt

# 计算曲线上的点的x和y坐标
# x = np.arange(0, 3 * np.pi, 0.1)
x = np.linspace(0, 3 * np.pi, 1000)
y_cos = np.cos(x)
y_sin = np.sin(x)

# 使用Matplotlib绘制点，并添加fmt和kwargs属性
plt.plot(x, y_cos, '-',
         label='Cosine Wave',  # 图例标签
         linewidth=1,  # 线宽
         color='blue',  # 线的颜色
         # marker='o',  # 标记样式
         # markersize=5,  # 标记的大小
         # markeredgecolor='black',  # 标记边缘的颜色
         # markeredgewidth=1,  # 标记边缘的宽度
         # markerfacecolor='none',  # 标记内部的颜色
         alpha=1 # 透明度
         )

plt.plot(x, y_sin, '--',
         label='Sine Wave',  # 图例标签
         linewidth=1,  # 线宽
         color='red',  # 线的颜色
         # marker='o',  # 标记样式
         # markersize=5,  # 标记的大小
         # markeredgecolor='black',  # 标记边缘的颜色
         # markeredgewidth=1,  # 标记边缘的宽度
         # markerfacecolor='none',  # 标记内部的颜色
         alpha=1 # 透明度
         )



# 显示图例
plt.legend()

# 显示图形
plt.show()
