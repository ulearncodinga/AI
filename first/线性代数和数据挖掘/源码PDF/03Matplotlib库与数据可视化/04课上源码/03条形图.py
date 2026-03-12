import matplotlib.pyplot as plt

# 数据
labels = [10, 15, 20, 25, 30]
values = [10, 15, 20, 25, 30]

# 创建条形图，并添加关键字参数
plt.bar(labels, values,
        width=0.7,              # 条形的宽度
        color='b',             # 条形的填充颜色
        # edgecolor='r',         # 条形边缘的颜色
        # linewidth=2,           # 条形边缘的线宽
        # linestyle='--',        # 条形边缘的线型
        alpha=0.7,             # 条形的透明度
        # hatch='+',            # 条形的填充图案
        align='center',        # 条形与x位置的对齐方式
        label='示例'         # 为条形创建图例时使用的标签
       )

# 显示图例
plt.legend()

# 显示图表
plt.show()