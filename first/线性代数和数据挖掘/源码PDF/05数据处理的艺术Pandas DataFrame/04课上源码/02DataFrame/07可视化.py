#
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # 创建一个 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, 4, 5],
#     'B': [10, 20, 30, 40, 50]
# })
#
# print(df)
#
# # 绘制线图
# df.plot(kind='bar')
#
# # 显示图像
# plt.show()




import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)

# 创建一个包含正态分布数据的 DataFrame
df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])


print(df)

# # 绘制 A 列的直方图，设置 20 个柱子，并指定图形大小
# df['A'].hist(bins=20, figsize=(8, 4))

# # 绘制所有列的直方图，每个直方图在不同的子图中
# df.hist(figsize=(10, 7), bins=30)

# 使用 by 参数按列 B 的值分组绘制列 A 的直方图
df.hist(column='A', by='B', bins=15, figsize=(10, 5))

# 显示图像
plt.show()