

#
# import pandas as pd
# import numpy as np
#
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 2, 3, 3, 3],
#     'B': ['a', 'b', 'b', 'c', 'c', 'c'],
#     'C': [np.nan, np.nan, 1, 2, 2, 2]
# })
#
# print(df)
#
# # 使用nunique()方法计算每列的唯一值数量
# unique_counts = df.nunique(axis=1, dropna=False)
#
# print(unique_counts)




# import pandas as pd
# import numpy as np
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 2, 3, 3, 3],
#     'B': ['a', 'b', 'b', 'c', 'c', 'c'],
#     'C': [np.nan, np.nan, 1, 2, 2, 2]
# })
#
# print(df)
#
# # 使用value_counts()方法计算值的频率
# value_counts = df.value_counts(subset=['C'], normalize=True, sort=True, ascending=True, dropna=False)
#
# print(value_counts)



# import pandas as pd
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 2, 3, 3, 3],
#     'B': [10, 20, 20.0, 30, 30, 30],
#     'C': ['foo', 'bar', 'bar', 'baz', 'baz', 'baz']
# })
#
# print(df)
#
# # 使用describe()方法生成统计摘要
# description = df.describe(percentiles=[0.5, 0.75, 0.9], include=['number'])
#
# print(description)


# import pandas as pd
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6]
# })
#
# # 创建深拷贝
# df_deep_copy = df.copy()
#
# # 创建浅拷贝
# df_shallow_copy = df.copy(deep=False)
#
# # 修改深拷贝中的数据
# df_deep_copy['A'][0] = 999
#
# # 修改浅拷贝中的数据
# df_shallow_copy['B'][0] = 888
#
# # 打印原始DataFrame和副本
# print("Original DataFrame:")
# print(df)
# print("\nDeep Copy:")
# print(df_deep_copy)
# print("\nShallow Copy:")
# print(df_shallow_copy)



# import pandas as pd
#
# # 创建两个列表，它们将用作多级索引的级别
# arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],  # 第一级索引的值
#           ['Captive', 'Wild', 'Captive', 'Wild']]   # 第二级索引的值
#
# # 使用arrays列表创建一个DataFrame，'Max Speed'列包含对应于多级索引的数据
# df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},  # 数据
#                   index=arrays)  # 使用arrays列表作为多级索引
#
# print(df)
# # # 重置所有索引级别，并将它们作为列添加到DataFrame中
# # df_reset = df.reset_index()
# # print(df_reset)
#
#
# # # 只重置第二级索引，并将它作为列添加到DataFrame中
# # df_reset_level_1 = df.reset_index(level=1)
# # print(df_reset_level_1)
#
# # 重置所有索引级别，但不将它们作为列添加到DataFrame中
# df_reset_drop = df.reset_index(drop=True)
# print(df_reset_drop)

#
#
# import pandas as pd
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, None],
#     'B': ['a', 'b', 'c', 'd'],
#     'C': [1.1, 2.2, 3.3, 4.4]
# })
#
# print(df)
#
# # 显示DataFrame的概要信息
# df.info(verbose=False, memory_usage=True, show_counts=True)




import pandas as pd

# 创建一个DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

print(df)

# 定义一个函数，计算每行的平均值
def mean_row(row):
    return row.mean()

# 应用函数到每行
result = df.apply(mean_row, axis=0, result_type='expand')

print(result)