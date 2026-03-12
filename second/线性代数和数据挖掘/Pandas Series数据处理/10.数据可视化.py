# Series.plot(*args,**kwargs)
'''kind : 图表类型，可以是以下之一
    'line' : 折线图（默认）
    'bar' : 柱状图
    'barh' : 水平柱状图
    'hist' : 直方图
    'box' : 箱线图
    'kde' : 核密度估计图
    'area' : 面积图
    'pie' : 饼图
    'scatter' : 散点图
    'hexbin' : 六边形箱图
ax : Matplotlib 轴对象，用于在指定的轴上绘制图表。如果不提供，则创建新的轴对象。
figsize : 图表的尺寸，格式为 (width, height) ，单位为英寸。
use_index : 是否使用 Series 的索引作为 x 轴标签。默认为 True 。
title : 图表的标题。
grid : 是否显示网格线。默认为 False 。
legend : 是否显示图例。默认为 False 。
xticks : x 轴的刻度位置。
yticks : y 轴的刻度位置。
xlim : x 轴的范围，格式为 (min, max) 。
ylim : y 轴的范围，格式为 (min, max) 。
color : 绘制颜色，可以是单个颜色或颜色列表。
label : 图例标签'''

# import pandas as pd
# import matplotlib.pyplot as plt
# s = pd.Series([1, 3, 2, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
# s.plot(kind='line', title='Line Plot', grid=True, figsize=(8, 4), style='r--',
# linewidth=2)
# plt.show()




'''hist
用于绘制Series数据直方图的方法。这个方法提供了多种参数来定制直方图的外观和样式。
Series.hist(by=None, ax=None, grid=True, xlabelsize=None, xrot=None, 
ylabelsize=None, yrot=None, figsize=None, bins=10, backend=None, legend=False, 
**kwargs)
by : 如果不是None，则将数据分组并分别绘制每个组的直方图。
ax : matplotlib的Axes对象，如果指定了，则直方图将绘制在该Axes上。
grid : 布尔值，默认为True，表示是否在直方图上显示网格线。
xlabelsize : int或str，用于设置x轴标签的字体大小。
xrot : int或float，用于设置x轴标签的旋转角度。
ylabelsize : int或str，用于设置y轴标签的字体大小。
yrot : int或float，用于设置y轴标签的旋转角度。
figsize : 元组，用于设置直方图的大小，格式为 (width, height)。
bins : int或序列，用于设置直方图的柱子数量或具体的边界。
backend : 用于指定绘图后端，通常Pandas会使用matplotlib。
legend : 布尔值，默认为False，表示是否在直方图上显示图例。
**kwargs : 其他关键字参数，将被传递给matplotlib的 hist 函数。
'''


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# 创建一个包含随机数据的Series
s = pd.Series(np.random.randn(1000))
# 绘制直方图
s.hist(bins=30, figsize=(8, 6), color='blue', edgecolor='black')
# 设置标题和轴标签
plt.title('Histogram of a Random Sample')
plt.xlabel('Value')
plt.ylabel('Frequency')
# 显示图表
plt.show()