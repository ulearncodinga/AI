'''
在 Matplotlib 中，Figure对象是整个绘图的顶级容器，它是所有绘图元素的基础，
提供了一个用于绘制图形的画布空间。
在 Matplotlib 中，Axes对象是进行数据绘制和设置坐标轴等操作的核心区域，它与
Figure对象紧密相关，共同构建了完整的绘图体系。


lt.figure(num=None, figsize=None, dpi=None, facecolor=None,
edgecolor=None, frameon=True, clear=False, **kwargs)


num : 可选参数，指定图形窗口的编号或名称。如果 num 为 None，则创建一个
新的图形窗口。如果 num 已经存在，则获取该图形窗口。
figsize : 可选参数，指定图形窗口的大小，以英寸为单位。默认为 None，使用
Matplotlib 的默认值。
dpi : 可选参数，指定图形的分辨率，以点每英寸为单位。默认为 None，使用
Matplotlib 的默认值。
facecolor : 可选参数，指定图形窗口的背景颜色。默认为 None，使用
Matplotlib 的默认值。
edgecolor : 可选参数，指定图形窗口的边缘颜色。默认为 None，使用
Matplotlib 的默认值。
frameon : 可选参数，布尔值，指定是否绘制图形窗口的边框。默认为 True。
clear : 可选参数，布尔值，指定是否清除图形窗口中的内容。默认为 False。
**kwargs : 其他关键字参数，用于进一步自定义图形窗口的属性。
'''
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot([0,1],[1,2])
plt.figure(num = 2)

# plt.plot([0,1],[1,3])
plt.show()