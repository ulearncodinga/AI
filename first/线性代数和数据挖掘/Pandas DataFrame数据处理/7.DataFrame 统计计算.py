# 1.统计 count 计算DataFrame中非Nan值的数量

# 0:竖直方向
# 1:水平方向
'''
DataFrame.count(axis=0, numeric_only=False)
axis : {0 或 ‘index’, 1 或 ‘columns’}，决定统计的方向。
如果 axis=0 或 axis='index' ，则对每列进行计数，返回一个 Series，其索引为列名，值
为每列非 NaN 值的数量。
如果 axis=1 或 axis='columns' ，则对每行进行计数，返回一个 Series，其索引为行索
引，值为每行非 NaN 值的数量。
numeric_only : 是否只计算数值列中的非 NaN 值的数量，忽略非数值列，默认为 False'''


import pandas as pd
import numpy as np
# 创建一个包含 NaN 值的 DataFrame
df = pd.DataFrame({
 'A': [1, 2, np.nan, 4],
 'B': [5, np.nan, np.nan, 8],
 'C': ['foo', 'bar', 'baz', np.nan]
})
# 计算每列非 NaN 值的数量
count_per_column = df.count()
print("Count per column:")
print(count_per_column)
# 计算每行非 NaN 值的数量
count_per_row = df.count(axis='columns')
print("\nCount per row:")
print(count_per_row)
# 只计算数值列的非 NaN 值的数量
count_numeric_only = df.count(numeric_only=True)
print("\nCount numeric only:")
print(count_numeric_only)







# sum
# 用于计算 DataFrame 中数值的总和。
# DataFrame.sum(axis=0, skipna=True, numeric_only=False, min_count=0, **kwargs)



import pandas as pd
import numpy as np
# 创建一个包含 NaN 值的 DataFrame
df = pd.DataFrame({
 'A': [1, 2, np.nan, 4],
 'B': [5, np.nan, np.nan, 8],
 'C': ['foo', 'bar', 'baz', 'qux']
})
# 计算每列的总和
sum_per_column = df.sum()
print("Sum per column:")
print(sum_per_column)
# 计算每行的总和
sum_per_row = df.sum(axis='columns')
print("\nSum per row:")
print(sum_per_row)
# 只计算数值列的总和
sum_numeric_only = df.sum(numeric_only=True)
print("\nSum numeric only:")
print(sum_numeric_only)
# 使用 min_count 参数
sum_with_min_count = df.sum(min_count=2)
print("\nSum with min_count=2:")
print(sum_with_min_count)