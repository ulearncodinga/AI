# add_subplot
'''
fig.add_subplot(nrows, ncols, plot_number)
'''

# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2,1,1)#两行一列第一行
# ax1.plot([0,1],[1,0])#两个点所构成的线段
#
# ax2= fig.add_subplot(2,1,2)
# ax2.plot([0,1],[1,0])
# plt.show()

# add_axes  : 任意位置添加子图
# matplotlib.figure.Figure.add_axes(rect, **kwargs)
'''
rect：一个形如 [左边界, 底边界, 宽度, 高度] 的列表，指定子图的位置和大
小。这些值是在归一化坐标中的，其中 0 表示图形的左边界，1 表示右边界，0 
表示底部，1 表示顶部。
**kwargs：其他可选的关键字参数，用于控制子图的属性。
'''
# import matplotlib.pyplot as plt
# # 创建一个图形
# fig = plt.figure()
# # 添加一个子图，位置是图形左下角的1/4处，大小为图形宽度的1/2和高度的1/2
# ax = fig.add_axes([0.1, 0.1, 0.5, 0.5])
# # 使用这个子图绘制一些数据
# ax.plot([0, 1], [0, 1])
# # 显示图形
# plt.show()

'''
该函数用于在图形（ Figure）上添加一个中心对齐的标题，这个标题位于所有子图
（Axes）的顶部。使用 suptitle 可以为整个图形设置一个总标题，而不仅仅是单
个子图。
'''

# matplotlib.figure.Figure.suptitle(t, **kwargs)
# import matplotlib.pyplot as plt
# # 创建一个图形
# fig = plt.figure()
# # 在图形上添加两个子图
# ax1 = fig.add_subplot(211) # 第一个子图，两行一列的第一个位置
# ax2 = fig.add_subplot(212) # 第二个子图，两行一列的第二个位置
# # 使用第一个子图绘制一些数据
# ax1.plot([0, 1], [0, 1])
# ax1.set_title('small title:ax1')
# # 使用第二个子图绘制一些数据
# ax2.plot([0, 1], [1, 0])
# ax2.set_title('small title :ax2')
# # 为整个图形添加一个总标题
# fig.suptitle('total title : Test Suptitle', fontsize=16, color='red')
# # 显示图形
# plt.show()







# text : 该函数用于在图表的任意位置添加文本，通常在 Axes 对象上调用，用于在绘图区域
# 的指定坐标位置上放置文本。
'''
matplotlib.axes.Axes.text(x, y, s, **kwargs)

x, y：浮点数或数组，指定文本的位置坐标。
s：字符串，要添加的文本内容。
**kwargs：其他可选的关键字参数，用于控制文本的样式，如 fontsize（字
体大小）、color（颜色）、 horizontalalignment（水平对齐方式，如 ‘left’, 
‘center’, ‘right’）、 verticalalignment（垂直对齐方式，如 ‘top’, ‘center’, 
‘bottom’）等。
'''
# import matplotlib.pyplot as plt
# # 创建一个图形和一个子图
# fig, ax = plt.subplots()
# # 绘制一些数据
# ax.plot([0, 1, 2], [0, 1, 0])
# # 在坐标 (1, 0.5) 处添加文本
# ax.text(1, 0.5, 'Sample Text', fontsize=12, color='red',
#         horizontalalignment='right', verticalalignment='center')
# # 显示图形
# plt.show()

# axes
'''
'''
# import matplotlib.pyplot as plt
# # 创建一个图形和一个子图
# fig, ax = plt.subplots(2, 2)
# # 获取图形中的所有轴
# axes = fig.axes
# # axes 是一个包含所有轴的列表
# for ax in axes:
# # 包含了每一个轴的x和y的相对位置(从左下角开始计算)、轴的宽度和高度(相对于整个图形)
#     print(ax)
# # 显示图像
# plt.show()


#get_facecolor
'''
该函数是一个用于获取轴（Axes）或填充区域（Patch）背景色的方法。具体来说，
它返回一个表示颜色的 RGBA 元组，其中 RGBA 分别代表

红色、绿色、蓝色和alpha（透明度）值。

这些值通常在 0 到 1 的范围内。
'''
# import matplotlib.pyplot as plt
# # 创建一个图形
# fig = plt.figure(figsize=(8, 6), facecolor='green')
# # 获取背景色
# facecolor = fig.get_facecolor()
# print(facecolor)
# # 显示图像
# plt.show()



'''
dpi
用于获取图像Figure对象的分辨率
'''

# import matplotlib.pyplot as plt
# 创建一个图形对象
# fig = plt.figure(dpi=300)


#获取图像的DPI
# dpi = fig.get_dpi()
# print(f"The DPI of the figure is:{dpi}")
#
# plt.show()


'''
get_gca
get current axis 缩写  获取当前图像中的当前坐标轴的对象
'''

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot([1,2,3],[1,2,3])
#
# ax = plt.gca()
#
# ax.set_title('Sample Plot')
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')

# plt.show()




'''get_label
该方法用于获取坐标轴（例如，线和柱状图）的标签。这个方法通常用于获取已经设
置好的标签，以便于后续的检查或修改。'''

# import matplotlib.pyplot as plt
# x = [1,2,3]
# y = [1,2,3]
# line, = plt.plot(x,y,label='Line 1')
#
# label = line.get_label()
# print('Label of the line:',label)
#
# plt.legend()
# plt.show()

'''

get_size_inches
该方法用于获取图形的大小，以英寸为单位'''

# import matplotlib.pyplot as plt
# # 创建一个图形对象，可以指定大小（宽度，高度）单位为英寸
# fig = plt.figure(figsize=(8, 6))
# # 使用 get_size_inches() 方法获取图形的大小
# size_in_inches = fig.get_size_inches()
# print('Size of the figure in inches:', size_in_inches)
# # 绘制一些数据
# plt.plot([1, 2, 3], [1, 2, 3])
# # 显示图形
# plt.show()
'''
set_size_inches
该方法用于设置图形对象的大小，单位为英寸。这个方法允许用户在创建图形后改变
其尺寸，从而影响输出图像的分辨率和显示效果。
'''
# import matplotlib.pyplot as plt
# # 创建一个图形对象
# fig = plt.figure(figsize=(7, 1))
# # 使用 set_size_inches 方法设置图形的大小
# fig.set_size_inches(10, 6) # 设置宽度为10英寸，高度为6英寸
# # 绘制一些数据
# plt.plot([1, 2, 3], [1, 2, 3])
# # 显示图形
# plt.show()



'''
tight_layout
该函数用于自动调整子图参数，使之填充整个图像区域，同时确保子图之间的标签和
标题不会重叠。这个方法对于创建布局整齐的图形非常有用，尤其是在子图较多或标
签较长时。
'''
# import matplotlib.pyplot as plt
# # 创建一个包含多个子图的图形
# fig, axs = plt.subplots(2, 2)
# # 在每个子图中绘制一些数据
# axs[0, 0].plot([1, 2, 3], [1, 2, 3])
# axs[0, 1].plot([1, 2, 3], [3, 2, 1])
# axs[1, 0].plot([1, 2, 3], [1, 3, 2])
# axs[1, 1].plot([1, 2, 3], [2, 1, 3])
# # 使用 tight_layout 自动调整子图布局
# fig.tight_layout()
# # 显示图形
# plt.show()





'''
subplots_adjust

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
wspace=None, hspace=None)

left : 子图区域左边缘与图形左边缘之间的距离，范围从 0 到 1
bottom : 子图区域下边缘与图形下边缘之间的距离，范围从 0 到 1。
right : 子图区域右边缘与图形右边缘之间的距离，范围从 0 到 1。
top : 子图区域上边缘与图形上边缘之间的距离，范围从 0 到 1。
wspace : 子图之间的水平间距，范围从 0 到 1。
hspace : 子图之间的垂直间距，范围从 0 到 1。
'''

import matplotlib.pyplot as plt

fig,axs=plt.subplots(2,2)
axs[0,0].plot([1,2,3],[1,2,3])
axs[0,1].plot([1,2,3],[3,2,1])
axs[1,0].plot([1,2,3],[1,3,2])
axs[1,1].plot([1,2,3],[2,1,3])

fig.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,hspace=0.5,wspace=0.5)
plt.show()
