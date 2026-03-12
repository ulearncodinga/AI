

# # pyplot库下的clf()
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
# # 使用 tight_layout 自动调整子图布局
# fig.tight_layout()
#
# # # Axes对象的clear
# # axs[0, 0].clear()
#
# # pyplot库下的clf()
# plt.clf()
#
# # 显示图形
# plt.show()




# # plt.gcf()
# import matplotlib.pyplot as plt
#
# # 创建一个新的图形和子图
# fig, ax = plt.subplots()
# ax.plot([1, 2, 3], [1, 2, 3])
#
# # 获取当前活动的图形对象
# current_fig = plt.gcf()
#
# print(current_fig is fig)
#
# # 设置图形的大小
# current_fig.set_size_inches(8, 6)
#
# # 设置图形的标题
# current_fig.suptitle('Example Plot')
#
# # 显示图形
# plt.show()



# # savefig
# import matplotlib.pyplot as plt
#
# # 创建一些数据
# x = [0, 1, 2, 3, 4]
# y = [0, 1, 4, 9, 16]
#
# # 绘制图形
# plt.plot(x, y)
#
# # 保存图形到文件
# plt.savefig('plot.png', dpi=200, bbox_inches='tight', facecolor='g')
#
# # 显示图形（可选）
# plt.show()


# # imshow
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 创建两个简单的数据集
# data1 = np.random.rand(10, 10)
# data2 = np.random.rand(10, 10)
#
# # 创建一个包含两个子图的图形
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# # 在第一个子图上显示第一个图像，并应用 viridis 颜色映射
# ax1.imshow(data1, cmap='viridis', aspect='equal', origin='lower')
# ax1.set_title('Image 1')
#
# # 在第二个子图上显示第二个图像，并应用 plasma 颜色映射
# ax2.imshow(data2, cmap='gray')
# ax2.set_title('Image 2')
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图像
# plt.show()




# # close
# import matplotlib.pyplot as plt
# import time
# # 创建并显示第一个图形
# plt.figure()
# plt.plot([1, 2, 3], [1, 2, 3])
# plt.show(block=False)
#
# # 创建并显示第二个图形
# plt.figure()
# plt.plot([3, 2, 1], [1, 2, 3])
# plt.show(block=False)
#
# # # 关闭当前图形
# # plt.close(1)
#
#
#
# # 创建第三个图形，但不显示
# plt.figure()
# plt.plot([1, 2, 3], [3, 2, 1])
# plt.show(block=False)
#
# time.sleep(2)
# # 关闭所有图形
# plt.close(fig='all')
#
#
# time.sleep(2)
# print(123)




# pause
import matplotlib.pyplot as plt
import numpy as np

# 创建一个图形和一个子图
fig, ax = plt.subplots()

# 生成一些数据
t = np.arange(0, 10, 0.01)
s = np.sin(t)

# 绘制第一条线
ax.plot(t, s)

# 显示图形
plt.show(block=False)

# 更新线的数据并暂停
for phase in np.arange(0, 2 * np.pi, 0.05):
    ax.plot(t + phase, np.sin(t + phase))
    plt.pause(0.01)

# 关闭图形
plt.close()

