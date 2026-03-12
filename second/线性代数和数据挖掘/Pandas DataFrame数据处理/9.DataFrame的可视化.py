# plot
"""DataFrame.plot(*args, **kwargs)"""
# import pandas as pd
# import matplotlib.pyplot as plt
# # 创建一个 DataFrame
# df = pd.DataFrame({
#  'A': [1, 2, 3, 4, 5],
#  'B': [10, 20, 30, 40, 50]
# })
# # 绘制线图
# df.plot(kind='line')
# # 显示图像
# plt.show()




# hist
"""DataFrame.hist(column=None, by=None, grid=True, xlabelsize=None, xrot=None, 
ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=None, 
layout=None, bins=10, backend=None, legend=False, **kwargs)

column : 字符串或字符串列表，可选。用于指定要绘制直方图的列。如果未指定，则默认绘制所有
数值列的直方图。
by : 字符串或字符串列表，可选。用于指定分组变量，以便为每个组绘制直方图。
grid : 布尔值，默认为 True。是否在直方图上显示网格线。
xlabelsize : 整数，可选。用于设置 x 轴标签的大小。
xrot : 浮点数，可选。用于设置 x 轴标签的旋转角度。
ylabelsize : 整数，可选。用于设置 y 轴标签的大小。
yrot : 浮点数，可选。用于设置 y 轴标签的旋转角度。
ax : Matplotlib axes 对象或数组，可选。如果提供，则在指定的 axes 对象上绘制直方图。
sharex : 布尔值，默认为 False。如果为 True，并且绘制多个直方图，则所有直方图共享相同的 x 
轴。
sharey : 布尔值，默认为 False。如果为 True，并且绘制多个直方图，则所有直方图共享相同的 y 
轴。
figsize : 元组（宽度，高度），可选。用于设置整个图形的大小。
layout : 元组（行数，列数），可选。用于指定直方图的布局。
bins : 整数或序列，默认为 10。用于设置直方图的柱子数量或边界。
backend : 字符串，可选。用于指定绘图的后端。默认情况下，Pandas 使用 Matplotlib。
legend : 布尔值，默认为 False。如果为 True，并且指定了 by 参数，则在直方图上显示图例。
**kwargs : 关键字参数，这些参数会被传递给 Matplotlib 的 hist 函数，用于进一步自定义直方
图。
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 创建一个包含正态分布数据的 DataFrame
df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])
# 绘制 A 列的直方图，设置 20 个柱子，并指定图形大小
df['A'].hist(bins=20, figsize=(8, 4))
# 绘制所有列的直方图，每个直方图在不同的子图中
df.hist(figsize=(10, 7), bins=30)
# 使用 by 参数按列 B 的值分组绘制列 A 的直方图
df.hist(column='A', by='B', bins=10, figsize=(100, 5))
# 显示图像
plt.tight_layout()#自适应
plt.show()