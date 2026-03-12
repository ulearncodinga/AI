import numpy as np
# # groupby
# import pandas as pd
# import numpy as np
#
#
# # 创建一个 DataFrame
# df = pd.DataFrame({
#     'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
#     'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
#     'C': [1, 3, 2, 5, 4, 1, 2, 3],
#     'D': [2, 5, 3, 7, 6, 2, 4, 6]
# })
#
# print(df)
#
# # 按 'A' 和 'B' 列分组，并计算每组的平均值
# grouped = df.groupby(['A', 'B']).mean()
#
# print(grouped)



# import pandas as pd
# import numpy as np
#
# # 创建两个列表，它们将用作多级索引的级别
# arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],  # 第一级索引的值
#           ['Captive', 'Wild', 'Captive', 'Wild']]   # 第二级索引的值
#
# # 使用arrays列表创建一个DataFrame，'Max Speed'列包含对应于多级索引的数据
# df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},  # 数据
#                   index=arrays)  # 使用arrays列表作为多级索引
#
# # 打印原始DataFrame，以查看其结构和数据
# print(df)
#
# # 使用groupby方法按第一级索引（即'arrays'列表的第一个元素）分组
# # level=0表示按照多级索引的第一级进行分组
# res = df.groupby(level=1).mean()  # 计算每个组（即每个不同的第一级索引值）的平均速度
#
# # 打印分组后的平均值，显示每个动物的Max Speed的平均值
# print(res)


# transform
# import pandas as pd
# import numpy as np
#
# # 创建一个 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, 4, 5],
#     'B': [10, 20, 30, 40, 50],
#     'C': [100, 200, 300, 400, 500]
# })
#
# print(df)
#
#
# def square(x):
#     return x ** 2
#
#
# # 调用transform函数，并将自定义函数带入
# transformed_df = df.transform(lambda x: x + x.sum(), axis=1)
#
# print(transformed_df)



# agg
import pandas as pd

# 创建一个 DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

print(df)

# # 应用单个聚合函数，计算每列的平均值
# result = df.agg('mean')
# print(result)

# # 对列 'A' 应用 'sum' 函数，对列 'B' 应用 'mean' 函数
# result = df.agg({'A': 'sum', 'B': 'mean'})
# print(result)

# # 对所有列应用多个聚合函数
# result = df.agg(['min', 'max', 'sum'])
# print(result)

# 定义一个自定义函数
def custom_agg(x):
    return (x.max() - x.min()) / x.std()

# 使用自定义函数进行聚合
result = df.agg(custom_agg)
print(result)