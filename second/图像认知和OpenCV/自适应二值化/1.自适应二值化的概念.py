import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成1000个符合正态分布的随机数
series = pd.Series(np.random.randn(1000))

# 绘制直方图
plt.figure(figsize=(8, 5))
series.hist(bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('正态分布数据直方图', fontsize=12)
plt.xlabel('数值', fontsize=10)
plt.ylabel('频率', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.show()
'''
'''
'''
自适应二值化的特点是能根据图像不同区域的局部特征来确定二值化的阈值。
'''