
# add_subplot
# import matplotlib.pyplot as plt
#
# # 创建一个图形窗口
# fig = plt.figure()
#
# # 添加第一个子图，位于2行1列布局中的第1个位置
# ax1 = fig.add_subplot(2, 1, 1)
# ax1.plot([0, 1], [0, 1])
# ax1.set_title('00')
#
# # 添加第二个子图，位于2行1列布局中的第2个位置
# ax2 = fig.add_subplot(2, 1, 2)
# ax2.plot([0, 1], [1, 0])
# ax2.set_title('01')
#
# plt.tight_layout()
#
# # 显示图形
# plt.show()


# add_axes

# import matplotlib.pyplot as plt
#
# # 创建一个图形
# fig = plt.figure()
#
# # 添加一个子图，位置是图形左下角的1/4处，大小为图形宽度的1/2和高度的1/2
# ax = fig.add_axes([0.5, 0.1, 0.5, 0.5])
#
# # 使用这个子图绘制一些数据
# ax.plot([0, 1], [0, 1])
#
# # 显示图形
# plt.show()


# # suptitle
# import matplotlib.pyplot as plt
#
# # 创建一个图形
# fig = plt.figure()
#
# # 在图形上添加两个子图
# ax1 = fig.add_subplot(121)  # 第一个子图，两行一列的第一个位置
# ax2 = fig.add_subplot(122)  # 第二个子图，两行一列的第二个位置
#
# # 使用第一个子图绘制一些数据
# ax1.plot([0, 1], [0, 1])
# ax1.set_title('ax1')
#
#
# # 使用第二个子图绘制一些数据
# ax2.plot([0, 1], [1, 0])
# ax2.set_title('ax2')
# # 为整个图形添加一个总标题
# fig.suptitle('Test Suptitle', fontsize=16, color='red')
#
# # 显示图形
# plt.show()


#
# # text
# import matplotlib.pyplot as plt
#
# # 创建一个图形和一个子图
# fig, ax = plt.subplots()
#
# # 绘制一些数据
# ax.plot([0, 1, 2], [0, 1, 0])
#
# # 在坐标 (1, 0.5) 处添加文本
# ax.text(1, 0.5, 'Sample Text', fontsize=12, color='red',
#         horizontalalignment='right', verticalalignment='top')
#
# # 显示图形
# plt.show()



# # axes
# import matplotlib.pyplot as plt
#
# # 创建一个图形和一个子图
# fig, ax = plt.subplots(2, 2)
#
# # 获取图形中的所有轴
# axes = fig.axes
#
#
# # axes 是一个包含所有轴的列表
# for ax in axes:
#     # 包含了每一个轴的x和y的相对位置(从左下角开始计算)、轴的宽度和高度(相对于整个图形)
#     print(ax)
#
# # 显示图像
# plt.show()
#

# # get_facecolor
# import matplotlib.pyplot as plt
#
# # 创建一个图形
# fig = plt.figure(figsize=(8, 6), facecolor='blue')
#
# # 获取背景色
# facecolor = fig.get_facecolor()
# print(facecolor)
#
# # 显示图像
# plt.show()


# # get_dpi
# import matplotlib.pyplot as plt
#
# # 创建一个图形对象
# fig = plt.figure(dpi=120)
#
# # 获取图形的DPI
# dpi = fig.get_dpi()
# print(f"The DPI of the figure is: {dpi}")
#
# # 显示图像
# plt.show()

# # gca
# import matplotlib.pyplot as plt
#
# # 创建一个图形和坐标轴
# plt.figure()
# plt.plot([1, 2, 3], [1, 2, 3])
#
# # 获取当前坐标轴
# ax = plt.gca()
#
# # 使用坐标轴对象进行自定义
# ax.set_title('Sample Plot')
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
#
# # 显示图形
# plt.show()
#


# import matplotlib.pyplot as plt
#
# # 创建一些数据
# x = [1, 2, 3]
# y = [1, 2, 3]
#
# # 绘制线图并设置标签
# line, = plt.plot(x, y, label='Line 1')
#
# # 获取线的标签
# label = line.get_label()
# print('Label of the line:', label)
#
# # 显示图形和图例
# plt.legend()
# plt.show()

# get_size_inches
# import matplotlib.pyplot as plt
#
# # 创建一个图形对象，可以指定大小（宽度，高度）单位为英寸
# fig = plt.figure(figsize=(8.0, 6.0))
#
# # 使用 get_size_inches() 方法获取图形的大小
# size_in_inches = fig.get_size_inches()
# print('Size of the figure in inches:', size_in_inches)
#
# # 绘制一些数据
# plt.plot([1, 2, 3], [1, 2, 3])
#
# # 显示图形
# plt.show()


# # set_size_inches
# import matplotlib.pyplot as plt
#
# # 创建一个图形对象
# fig = plt.figure(figsize=(2, 2))
#
# # 使用 set_size_inches 方法设置图形的大小
# fig.set_size_inches(10, 6)  # 设置宽度为10英寸，高度为6英寸
#
# # 绘制一些数据
# plt.plot([1, 2, 3], [1, 2, 3])
#
# # 显示图形
# plt.show()


# # set_dpi
# import matplotlib.pyplot as plt
#
# # 创建一个图形对象
# fig, ax = plt.subplots()
#
# # 设置图形的尺寸为 6x4 英寸
# # fig.set_size_inches(10, 6)
#
# # 使用 set_dpi 方法设置图形的分辨率
# fig.set_dpi(200)  # 设置分辨率为 100 DPI
#
# # 绘制一些数据
# ax.plot([1, 2, 3], [1, 2, 3])
#
#
# print(fig.get_size_inches())
# print(fig.get_dpi())
#
# # 显示图形
# plt.show()



# # tight_layout
# import matplotlib.pyplot as plt
#
# # 创建一个包含多个子图的图形
# fig, axs = plt.subplots(2, 2)
#
# # 在每个子图中绘制一些数据
# axs[0, 0].plot([1, 2, 3], [1, 2, 3])
# axs[0, 0].set_title('00')
# axs[0, 0].set_xlabel('x')
# axs[0, 0].set_ylabel('y')
#
#
# axs[0, 1].plot([1, 2, 3], [3, 2, 1])
# axs[0, 1].set_title('01')
# axs[0, 1].set_xlabel('x')
# axs[0, 1].set_ylabel('y')
#
# axs[1, 0].plot([1, 2, 3], [1, 3, 2])
# axs[1, 0].set_title('10')
# axs[1, 0].set_xlabel('x')
# axs[1, 0].set_ylabel('y')
#
# axs[1, 1].plot([1, 2, 3], [2, 1, 3])
# axs[1, 1].set_title('11')
# axs[1, 1].set_xlabel('x')
# axs[1, 1].set_ylabel('y')
#
#
# # 使用 tight_layout 自动调整子图布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()


# # subplots_adjust
# import matplotlib.pyplot as plt
#
# # 创建一个包含多个子图的图形
# fig, axs = plt.subplots(2, 2)
#
# # 在每个子图中绘制一些数据
# axs[0, 0].plot([1, 2, 3], [1, 2, 3])
# axs[0, 1].plot([1, 2, 3], [3, 2, 1])
# axs[1, 0].plot([1, 2, 3], [1, 3, 2])
# axs[1, 1].plot([1, 2, 3], [2, 1, 3])
#
# # 使用 subplots_adjust 手动调整子图布局
# plt.subplots_adjust(left=0.4, right=0.5, top=0.9, bottom=0.1, hspace=0.2, wspace=0.2)
#
# # 显示图形
# plt.show()



# clear
import matplotlib.pyplot as plt

# 创建一个包含多个子图的图形
fig, axs = plt.subplots(2, 2)

# 在每个子图中绘制一些数据
axs[0, 0].plot([1, 2, 3], [1, 2, 3])
axs[0, 1].plot([1, 2, 3], [3, 2, 1])
axs[1, 0].plot([1, 2, 3], [1, 3, 2])
axs[1, 1].plot([1, 2, 3], [2, 1, 3])

# 使用 tight_layout 自动调整子图布局
fig.tight_layout()

# # Axes对象的clear
# axs[0, 0].clear()

# Figure对象的clear
fig.clear()

# 显示图形
plt.show()

