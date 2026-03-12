# 根据每一个子图定制位置和大小

'''
matplotlib.gridspec.GridSpec(nrows, ncols, figure=None, left=None,
bottom=None, right=None, top=None, wspace=None, hspace=None,
width_ratios=None, height_ratios=None)



nrows : 网格的行数。
ncols : 网格的列数。
figure : 可选，如果提供， GridSpec 将与这个指定的 Figure 实例关联。如果
没有提供，GridSpec 将与当前活动的 Figure 关联。
left , bottom , right , top : 可选，这些参数用于指定网格在整个画布中的位
置，它们的值应该在 0 到 1 之间，表示画布宽度和高度的百分比。例如，
left=0.1 表示网格左边界距离画布左边界 10% 的位置。
wspace : 可选，子图之间的水平间距，这个值是以画布宽度的百分比来表示的。
hspace : 可选，子图之间的垂直间距，这个值是以画布高度的百分比来表示的。
width_ratios : 可选，指定各列的相对宽度。默认情况下，所有列的宽度都是相
等的。如果提供这个参数，那么每一列的宽度将是其对应值的相对比例。
height_ratios : 可选，指定各行的相对高度。默认情况下，所有行的高度都是
相等的。如果提供这个参数，那么每一行的高度将是其对应值的相对比例。
'''



import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 6))
# 创建一个 2 行 3 列的 GridSpec，第一列宽度是其他列的两倍，第二行高度是其他行
gs = GridSpec(2, 3)
# 添加子图
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1:])
# 注意：gs[1, 1:] 表示第二行的第二列和第三列
plt.show()