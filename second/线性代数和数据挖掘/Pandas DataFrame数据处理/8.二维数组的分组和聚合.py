# grouppby
# 将数组分割成多个组
'''
DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True,
group_keys=True, observed=False, dropna=True)

by : 用于确定分组依据。
axis : {0 或 ‘index’, 1 或 ‘columns’}，默认为 0。这个参数决定了分组是在哪个轴上进行。
level : 如果轴是多索引，则按指定的索引级别分组。
as_index : 布尔值，默认为 True。当为 True 时，分组名称将作为结果的索引；如果为 False，则
结果会保持原有的 DataFrame 结构，分组名称会作为一个普通列出现。
sort : 布尔值，默认为 True。如果为 True，则对分组键进行排序；如果为 False，则不排序，保
持原有的分组顺序。
group_keys : 布尔值，默认为 True。如果为 True，则将组键添加到聚合后的结果中；如果为
False，则不添加。
observed : 布尔值，默认为 False。仅当分组依据是多索引且包含分类数据时适用，如果为
True，则仅显示分类数据中的观察值。
dropna : 布尔值，默认为 True。如果为 True，并且 by 是一个列表，则排除任何含有 NaN 的
组。
'''
# import pandas as pd
# # 创建一个 DataFrame
# df = pd.DataFrame({
#  'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
#  'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
#  'C': [1, 3, 2, 5, 4, 1, 2, 3],
#  'D': [2, 5, 3, 7, 6, 2, 4, 6]
# })
# # 按 'A' 和 'B' 列分组，并计算每组的平均值
# grouped = df.groupby(['A', 'B']).mean()
# print(grouped)



# eg:2

# import pandas as pd
# # 创建两个列表，它们将用作多级索引的级别
# arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'], # 第一级索引的值
#   ['Captive', 'Wild', 'Captive', 'Wild']]  # 第二级索引的值
# # 使用arrays列表创建一个DataFrame，'Max Speed'列包含对应于多级索引的数据
# df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]}, # 数据
#  index=arrays) # 使用arrays列表作为多级索引
# # 打印原始DataFrame，以查看其结构和数据
# print(df)
# # 使用groupby方法按第一级索引（即'arrays'列表的第一个元素）分组
# # level=0表示按照多级索引的第一级进行分组
# res = df.groupby(level=0).mean() # 计算每个组（即每个不同的第一级索引值）的平均速度
# # 打印分组后的平均值，显示每个动物的Max Speed的平均值
# print(res)



# transform
''' 对 DataFrame 或其中一个列应用一个函数，并返回一个与原始 DataFrame 具有相同索引的对象。此方
 法不会更改原始 DataFrame，而是返回一个新的 DataFrame，其中包含应用函数后的结果
 
 DataFrame.transform(func, axis=0, *args, **kwargs)
 
 func : 函数，应用于每个组或列。这个函数可以是一个内置函数，比如 numpy.mean ，也可以是一
个自定义函数。
axis : {0 或 ‘index’, 1 或 ‘columns’}，默认为 0。这个参数决定了函数是在哪个轴上应用：
如果 axis=0 或 axis='index' ，则函数按列应用。
如果 axis=1 或 axis='columns' ，则函数按行应用。
*args : 位置参数，将传递给 func 。
**kwargs : 关键字参数，将传递给 func 。'''

# import  pandas as pd
#
# import  numpy as np
# # 创建一个 DataFrame
# df = pd.DataFrame({
#  'A': [1, 2, 3, 4, 5],
#  'B': [10, 20, 30, 40, 50],
#  'C': [100, 200, 300, 400, 500]
# })
# def square(x):
#   return x ** 2
# # 调用transform函数，并将自定义函数带入
# transformed_df = df.transform(square)
# print(transformed_df)


# agg
# 它允许你应用一个或多个聚合函数到 DataFrame 的列或行上，并返回聚合后的结果。
'''
DataFrame.agg(func=None, axis=0, *args, **kwargs)

func : 函数或函数列表/字典，应用于 DataFrame 的列或行。如果传递一个函数，它将应用于所有
列。如果传递一个函数列表或字典，则可以分别对不同的列应用不同的函数。可以使用的函数包括
NumPy 的统计函数、Python 的内置函数、Pandas 的自定义聚合函数，或者自定义的 lambda 函
数。
axis : {0 或 ‘index’, 1 或 ‘columns’}，默认为 0。这个参数决定了聚合操作是在哪个轴上执行：
如果 axis=0 或 axis='index' ，则函数按列应用（逐行操作）。
如果 axis=1 或 axis='columns' ，则函数按行应用（逐列操作）。
*args : 位置参数，将传递给 func 。
**kwargs : 关键字参数，将传递给func

'''
import pandas as pd
# 创建一个 DataFrame
df = pd.DataFrame({
 'A': [1, 2, 3, 4, 5],
 'B': [10, 20, 30, 40, 50]
})
# 应用单个聚合函数，计算每列的平均值
result = df.agg('mean')
print(result)
# 对列 'A' 应用 'sum' 函数，对列 'B' 应用 'mean' 函数
result = df.agg({'A': 'sum', 'B': 'mean'})
print(result)
# 对所有列应用多个聚合函数
result = df.agg(['min', 'max', 'sum'])
print(result)
# 定义一个自定义函数
def custom_agg(x):
  return (x.max() - x.min()) / x.std()
# 使用自定义函数进行聚合
result = df.agg(custom_agg)
print(result)