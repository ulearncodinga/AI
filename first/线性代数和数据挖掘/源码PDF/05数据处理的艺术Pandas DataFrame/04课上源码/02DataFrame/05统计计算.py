# import pandas as pd
# import numpy as np
#
# # 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan, 4],
#     'B': [5, np.nan, np.nan, 8],
#     'C': ['foo', 'bar', 'baz', np.nan]
# })
#
# print(df)
#
# # 计算每列非 NaN 值的数量
# count_per_column = df.count()
# print("Count per column:")
# print(count_per_column)
#
# # 计算每行非 NaN 值的数量
# count_per_row = df.count(axis=1)
# print("\nCount per row:")
# print(count_per_row)
#
# # 只计算数值列的非 NaN 值的数量
# count_numeric_only = df.count(numeric_only=True)
# print("\nCount numeric only:")
# print(count_numeric_only)


#
# import pandas as pd
# import numpy as np
#
# # 创建一个包含 NaN 值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan, 4],
#     'B': [5, np.nan, np.nan, 8],
#     'C': ['foo', 'bar', 'baz', 'qux']
# })
#
# print(df)
#
# # 计算每列的总和
# sum_per_column = df.sum()
# print("Sum per column:")
# print(sum_per_column)
#
# # # 计算每行的总和
# # sum_per_row = df.sum(axis=1)
# # print("\nSum per row:")
# # print(sum_per_row)
#
# # 只计算数值列的总和
# sum_numeric_only = df.sum(numeric_only=True)
# print("\nSum numeric only:")
# print(sum_numeric_only)
#
# # 使用 min_count 参数
# sum_with_min_count = df.sum(min_count=3)
# print("\nSum with min_count=2:")
# print(sum_with_min_count)




import pandas as pd
import numpy as np

# 创建一个 DataFrame
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, 3, 2, 1],
})

print(df)

# 计算每列的累积最大值
cummax_per_column = df.cummax(axis=0)
print("Cumulative max per column:")
print(cummax_per_column)

# 计算每行的累积最大值
cummax_per_row = df.cummax(axis=1)
print("\nCumulative max per row:")
print(cummax_per_row)