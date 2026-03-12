'''
散点图
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None,
cmap=None, norm=None, vmin=None, vmax=None, alpha=None,
linewidths=None, edgecolors=None, **kwargs)

x , y : 数组或标量，代表散点图中每个点的 x 和 y 坐标。
s : 散点的大小。可以是一个标量或数组，用于指定每个点的大小。
c : 散点的颜色。可以是单个颜色，也可以是数组，为每个点指定颜色。如果提供
了 cmap 参数，则 c 可以是颜色映射的值。
marker : 散点的标记样式，例如 'o' 表示圆圈， 's' 表示正方形。
cmap : Colormap，用于将 c 中的数值映射到颜色。
norm : 用于标准化 c 的颜色数据的 Normalize 对象，确保数据在指定的范围内
映射到颜色映射。
vmin , vmax : 分别是 c 中数据的最小值和最大值，用于色彩映射。
alpha : 透明度，范围从 0（完全透明）到 1（完全不透明）。
linewidths : 散点边缘的线宽。
edgecolors : 散点边缘的颜色。
'''
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(0,3*np.pi,0.1)
# y = np.sin(x)
# colors = y
# plt.scatter(x,y,
#             s=10,              # 散点的大小
#             c=colors,          # 散点的颜色，这里使用y值映射颜色
#             marker='o',        # 散点的标记样式
#             cmap='viridis',    # 颜色映射
#             norm=None,         # 默认的标准化
#             vmin=-1,           # 颜色映射的最小值
#             vmax=1,            # 颜色映射的最大值
#             alpha=0.5,         # 透明度
#             linewidths=0.5,    # 散点边缘的线宽
#             edgecolors='w'     # 散点边缘的颜色
#             )
# plt.colorbar()
# plt.show()


# 条形图
'''
matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

x : 条形中心的x坐标。这个参数可以是单个值或者一个列表，表示每个条形的中
心位置。
height : 条形的高度。这通常是一个数值或者一个列表，表示每个条形的高度。
width : 条形的宽度，默认为0.8。这可以是单个值，用于所有条形，或者一个列
表，用于指定每个条形的宽度。
bottom : 条形的起始位置，默认为0。这可以是单个值或者一个列表，用于堆叠
条形图。
align : 条形的对齐方式，可以是’center’（中心对齐，默认），‘edge’（左对
齐）。
data : 如果提供了 data参数，则x和height参数可以指定为字符串，这些字符串
将被解释为pandas DataFrame列的名称。
**kwargs : 其他关键字参数，包括：
color 或 facecolor : 条形的填充颜色。
edgecolor 或 linewidth : 条形边缘的颜色和线宽。
linestyle : 条形边缘的线型，例如 '-' , '--' , '-.' , ':'。
alpha : 条形的透明度，范围从0（完全透明）到1（完全不透明）。
hatch : 条形的填充图案，例如 '/' , '\\' , '|' , '-' , '+' , 'x' , 'o' , 'O' , 
'*'。
picker : 控制条形是否可以被交互式选择。
log : 如果设置为True，则条形的高度将以对数尺度表示。
label : 为条形创建图例时使用的标签。
 
'''
# import matplotlib.pyplot as plt
#
# labels = ['A','B','C','D','E']
# values = [23,23,45,78,12]
# plt.bar(labels,values,
#         width=0.3,
#         color = 'b',  # 条形的填充颜色
#         edgecolor = 'r',  # 条形边缘的颜色
#         linewidth = 2,  # 条形边缘的线宽
#         linestyle = '-',  # 条形边缘的线型
#         alpha = 0.7,  # 条形的透明度
#         hatch = '-',  # 条形的填充图案
#         align = 'center',  # 条形与x位置的对齐方式
#         label = 'test'  # 为条形创建图例时使用的标签
#             )
# plt.legend()
# plt.show()





#饼图
'''
plt.pie(x, explode=None, labels=None, colors=None, autopct=None, 
startangle=0, shadow=False, radius=1, wedgeprops=None, 
textprops=None, center=(0, 0), frame=False, rotatelabels=False, 
**kwargs)



x : 饼图中每个扇形的尺寸。它应该是一个非负的数组或序列。
explode : 一个与 x 相同长度的数组，用于指定每个扇形是否突出显示。如果
explode[i] 不为 0，则第 i 个扇形将与其他扇形分离。
labels : 一个字符串列表，用于指定每个扇形的标签。
colors : 一个颜色列表，用于指定每个扇形的颜色。
autopct : 一个格式化字符串，用于在饼图上显示每个扇形的百分比。可以使用
%1.1f%% 来显示一个保留一位小数的百分比。
startangle : 饼图开始的角度，默认为 0（即从 x 轴正方向开始）。
shadow : 布尔值，用于指定是否为饼图添加阴影。
radius : 饼图的半径，默认为 1。
wedgeprops : 字典，用于指定饼图中每个扇形的属性，如边缘颜色和宽度。
textprops : 字典，用于指定饼图中标签的文本属性。
center : 一个元组，用于指定饼图的中心位置。
frame : 布尔值，用于指定是否为饼图添加一个框。
rotatelabels : 布尔值，用于指定是否旋转标签以适应扇形。
**kwargs : 其他关键字参数，包括：
normalize : 布尔值，默认为 True。如果 normalize 为 True，则 x 中的
值将被归一化以使它们的总和等于 1。如果 normalize 为 False，则 x 中
的值将直接用于绘制饼图，而不进行归一化处理。

hatch : 字符串或列表，用于设置饼图各个部分的阴影或图案填充。如果提供
一个字符串，则所有扇形都将使用相同的图案。如果提供一个列表，则每个
扇形可以有不同的图案。例如， hatch='/' 将为所有扇形添加斜线图案。
'''
import matplotlib.pyplot as plt
# 数据
sizes = [25, 35, 20, 20]
labels = ['A', 'B', 'C', 'D']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# 绘制饼图，使用 **kwargs 定制扇形
plt.pie(sizes,                          # 饼图中每个扇形的尺寸
        explode=[0, 0, 0, 0],           # 用于指定每个扇形是否突出显示
        labels=labels,                  # 用于指定每个扇形的标签
        colors=colors,                  # 用于指定每个扇形的颜色
        autopct='%1.1f%%',              # 用于在饼图上显示每个扇形的百分比
        startangle=140,                 # 饼图开始的角度，默认为 0（即从 x 轴正方向开始）
        shadow=False,                   # 用于指定是否为饼图添加阴影
        radius=1,                       # 饼图的半径
        wedgeprops=dict(edgecolor='black', linewidth=2, linestyle='-'),  # 指定饼图中每个扇形的属性
        textprops=dict(color='red', weight='bold'),  # 用于指定饼图中标签的文本属性，这里指定文本的颜色和字体的粗细
        center=(0, 0),                  # 用于指定饼图的中心位置
        frame=False,                     # 用于指定是否为饼图添加一个框
)
# 显示图表
plt.show()







