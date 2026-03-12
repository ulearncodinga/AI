'''
clf
与Figure对象的clear方法的作用类似，但clf归属于pyplot库，而不是Figure对象的方
法。
'''
# import matplotlib.pyplot as plt
# fig,axs = plt.subplots(2,2)
#
# axs[0,0].plot([1,2,3],[1,2,3])
# axs[0,1].plot([1,2,3],[3,2,1])
# axs[1, 0].plot([1, 2, 3], [1, 3, 2])
# axs[1, 1].plot([1, 2, 3], [2, 1, 3])
#
# # 使用 tight_layout 自动调整子图布局
# fig.tight_layout()
# # Axes对象的clear
# axs[0, 0].clear()
# # pyplot库下的clf()
# plt.clf()
# plt.show()

'''
gcf
该函数用于返回当前活动的图形（figure）对象。

matplotlib.pyplot.gcf()

返回当前活动的 Figure 对象。如果没有活动的图形，这个函数将创建一个新的
图形，并将其作为当前活动的图形返回。
'''
# import matplotlib.pyplot as plt
# # 创建一个新的图形和子图
# fig, ax = plt.subplots()
# ax.plot([1, 2, 3], [1, 2, 3])
# # 获取当前活动的图形对象
# current_fig = plt.gcf()
# # 设置图形的大小
# current_fig.set_size_inches(8, 6)
# # 设置图形的标题
# current_fig.suptitle('Example Plot')
# # 显示图形
# plt.show()


'''
savefig
该函数用于将当前图形保存到文件中。该函数通常在完成图形绘制后调用，以便将图
形输出为图像文件，便于保存或分享。


matplotlib.pyplot.savefig(fname, dpi='figure', format=None, 
metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', 
edgecolor='auto', backend=None, **kwargs)
'''

# import matplotlib.pyplot as plt
#
# x = [0,1,2,3,4,5]
# y = [0,1,4,9,16,25]
#
# plt.plot(x,y)
#
# plt.savefig('selfmake.png',dpi=150,bbox_inches='tight',facecolor='g')
#
# plt.show()


'''
imshow

matplotlib.pyplot.imshow(X, cmap=None, norm=None, aspect=None, 
interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, 
**kwargs)


X：图像数据，通常是二维数组，但也可以是一维数组（在这种情况下，数据将
被解释为线图）。数据类型通常是浮点数或整数。
cmap：颜色映射（colormap），用于指定图像中数值到颜色的映射。默认是
None，此时将使用默认的 colormap，通常是 viridis。
norm：用于标准化数据值的归一化对象，可以确保颜色映射是线性的或者使用
其他映射方式。
aspect：控制图像的纵横比。 'auto' 保持宽高比， 'equal' 使每个像素都是
正方形，或者可以是一个数字。
interpolation：指定图像缩放时的插值方法。常见的选项有 'nearest'（最近
邻插值）、'bilinear'（双线性插值）、 'bicubic'（双三次插值）等。
alpha：图像的透明度，取值范围是 [0, 1]。
vmin, vmax：用于颜色映射的数据值范围。默认情况下，这些值是从数据中自
动推断出来的。
origin：指定图像的原点， 'upper' 表示原点在左上角， 'lower' 表示原点在
左下角。
**kwargs：其他关键字参数。
'''
# import matplotlib.pyplot as plt
# import numpy as np
# # 创建两个简单的数据集
# data1 = np.random.rand(10, 10)
# data2 = np.random.rand(10, 10)
# # 创建一个包含两个子图的图形
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# # 在第一个子图上显示第一个图像，并应用 viridis 颜色映射
# ax1.imshow(data1, cmap='viridis')
# ax1.set_title('Image 1')
# # 在第二个子图上显示第二个图像，并应用 plasma 颜色映射
# ax2.imshow(data2, cmap='plasma')
# ax2.set_title('Image 2')
# # 调整子图之间的间距
# plt.tight_layout()
# # 显示图像
# plt.show()


'''

close

该函数用于关闭一个或多个打开的图形窗口。在 Matplotlib 中，每次调用绘图命令
（如 plt.plot() 或 plt.imshow()）时，都会创建一个新的图形（Figure）和一
个或多个子图（Axes）。 close 函数可以用来关闭这些图形窗口，释放资源，并允
许用户清理不再需要的图形

matplotlib.pyplot.close(fig=None)

fig ( Figure 对象或整数，可选）：指定要关闭的图形。如果是一个 Figure 对
象，则关闭该图形。如果是一个整数，则关闭与该编号对应的图形。如果省略或
为 None，则关闭当前图形。’all'表示关闭所有的图像。


'''
#
# import matplotlib.pyplot as plt
# # 创建并显示第一个图形
# plt.figure()
# plt.plot([1, 2, 3], [1, 2, 3])
# plt.show()
# # 创建并显示第二个图形
# plt.figure()
# plt.plot([3, 2, 1], [1, 2, 3])
#
# plt.show()
# # 关闭当前图形
# plt.close()
# # 创建第三个图形，但不显示
# plt.figure()
# plt.plot([1, 2, 3], [3, 2, 1])
# # 关闭所有图形
# plt.close(fig='all')
# plt.show()



'''
pause
用于在动画或交互式绘图过程中暂停一段时间。这个函数在动画制作或者需要在绘图
之间插入延迟时非常有用。

matplotlib.pyplot.pause(interval)

interval : 这个参数是一个浮点数，表示暂停的秒数。默认值通常是 0.1 秒。

'''
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
for phase in np.arange(0, 2 * np.pi, 0.1):
   ax.plot(t + phase, np.sin(t + phase))
   plt.pause(0.01)
# 关闭图形
plt.close()
