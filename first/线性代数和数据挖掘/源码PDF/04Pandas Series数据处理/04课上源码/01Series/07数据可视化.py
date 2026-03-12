

# plot
import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series([1, 3, 2, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
s.plot(kind='box', title='Line Plot', grid=False, figsize=(8, 4), style='r--',label='test', legend=True)
plt.show()



# hist
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 创建一个包含随机数据的Series
s = pd.Series(np.random.randn(1000))

# 绘制直方图
s.hist(bins=5, figsize=(8, 6), color='blue', edgecolor='black')

# 设置标题和轴标签
plt.title('Histogram of a Random Sample')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图表
plt.show()