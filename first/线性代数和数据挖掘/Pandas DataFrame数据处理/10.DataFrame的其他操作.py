'''
1 nunique
用于计算DataFrame中唯一值的数量。(就是不重复的值的数量,NAan不算)
'''
# DataFrame.nunique(axis=0, dropna=True)


# import pandas as pd
# import numpy as np
# # 创建一个DataFrame
# df = pd.DataFrame({
#  'A': [1, 2, 2, 3, 3, 3],
#  'B': ['a', 'b', 'b', 'c', 'c', 'c'],
#  'C': [np.nan, np.nan, 1, 2, 2, 2]
# })
# # 使用nunique()方法计算每列的唯一值数量
# unique_counts = df.nunique(axis=0, dropna=True)
# print(unique_counts)




# 2.value_counts 计算DataFrame中各个值出现的频率的一个方法
'''
DataFrame.value_counts(subset=None, normalize=False, sort=True, ascending=False, 
dropna=True)
'''
# import pandas as pd
# import numpy as np
# # 创建一个DataFrame
# df = pd.DataFrame({
#  'A': [1, 2, 4, 3, 5, 3],
#  'B': ['a', 'b', 'b', 'l', 'c', 'c'],
#  'C': [np.nan, np.nan, 1, 2, 2, 2]
# })
# # 使用value_counts()方法计算值的频率
# value_counts = df.value_counts(subset=['A', 'B'], normalize=False, sort=True,
# ascending=False, dropna=True)
# print(value_counts)


'''
describe
用于生成DataFrame中数值列的统计摘要，会返回一个DataFrame，其中包含以下统计信息：
count ：非NA值的数量
mean ：平均值
std ：标准差
min ：最小值
分位数 ：可指定
max ：最大值

DataFrame.describe(percentiles=None, include=None, exclude=None)

percentiles=None ：一个列表形式的数值，用于指定要计算的分位数。默认情况下，会计算
25%，50%，和75%这三个分位数。如果你想要不同的分位数，可以在这里指定。
include=None ：一个列表形式的字符串或字符串序列，用于指定要包含在描述统计中的数据类
型。默认情况下，只包括数值类型的数据。
exclude=None ：一个列表形式的字符串或字符串序列，用于指定要从描述统计中排除的数据类
型。

'''
# import pandas as pd
# # 创建一个DataFrame
# df = pd.DataFrame({
#  'A': [1, 2, 2, 3, 3, 3],
#  'B': [10, 20, 20, 30, 30, 30],
#  'C': ['foo', 'bar', 'bar', 'baz', 'baz', 'baz']
# })
# # 使用describe()方法生成统计摘要
# description = df.describe(percentiles=[0.5, 0.75, 0.9], include=['number'],
# exclude=['object'])
# print(description)

'''
copy(浅拷贝会影响原数据,深拷贝不会影响原来的数据)
用于创建DataFrame的一个副本。

DataFrame.copy(deep=True)

deep=True ：默认为True，表示创建一个深拷贝。在深拷贝中，DataFrame中的所有数据都会被
复制到新的DataFrame中，因此原始DataFrame中的数据与副本DataFrame中的数据是完全独立
的。如果你对副本进行修改，原始DataFrame不会受到影响，反之亦然。
'''
# import pandas as pd
#
# df = pd.DataFrame({
#  'A': [1, 2, 3],
#  'B': [4, 5, 6]
# })
# # 创建深拷贝
# df_deep_copy = df.copy()
# # 创建浅拷贝
# df_shallow_copy = df.copy(deep=False)
# # 修改深拷贝中的数据
# df_deep_copy.loc[0,'A'] = 999
# # 修改浅拷贝中的数据
# df_shallow_copy.loc[0,'B'] = 888
# # 打印原始DataFrame和副本
# print("Original DataFrame:")
# print(df)
# print("\nDeep Copy:")
# print(df_deep_copy)
# print("\nShallow Copy:")
# print(df_shallow_copy)


# reset_index
# 用于重置DataFrame的索引。

'''
DataFrame.reset_index(level=None, *, drop=False, inplace=False, col_level=0, 
col_fill='', allow_duplicates=_NoDefault.no_default, names=None)

level=None ：整数或索引名称，用于指定要重置的索引级别。对于多级索引的DataFrame，可以
指定一个级别或级别列表。如果为None，则重置所有级别。
drop=False ：布尔值，默认为False。如果为True，则重置索引后，原始索引将不会作为列添加
到DataFrame中。如果为False，则原始索引将作为新列添加到DataFrame中。
inplace=False ：布尔值，默认为False。如果为True，则直接在原始DataFrame上进行操作，不
返回新的DataFrame。
col_level=0 ：如果原始索引是多级索引，则指定新列的索引级别。
col_fill='' ：如果原始索引是多级索引，并且 col_level 比原始索引的级别多，则使用此值填
充缺失的级别。
allow_duplicates=_NoDefault.no_default ：允许索引列中出现重复值。如果设置为False，
则在尝试插入重复列时会引发ValueError。
names=None ：列表或元组，用于为新列指定名称。默认情况下，如果 drop 为False，则使用原始
索引的名称。

'''
# import pandas as pd
# # 创建两个列表，它们将用作多级索引的级别
# arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'], # 第一级索引的值
#   ['Captive', 'Wild', 'Captive', 'Wild']] # 第二级索引的值
# # 使用arrays列表创建一个DataFrame，'Max Speed'列包含对应于多级索引的数据
# df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]}, # 数据
# index=arrays) # 使用arrays列表作为多级索引
# print(df)
# # 重置所有索引级别，并将它们作为列添加到DataFrame中
# df_reset = df.reset_index()
# print(df_reset)
# # 只重置第二级索引，并将它作为列添加到DataFrame中
# df_reset_level_1 = df.reset_index(level=1)
# print(df_reset_level_1)
# # 重置所有索引级别，但不将它们作为列添加到DataFrame中
# df_reset_drop = df.reset_index(drop=True)
# print(df_reset_drop)

'''
info
用于显示DataFrame的概要信息的一个便捷方法，它提供了关于DataFrame的列、非空值数量、数据类
型以及内存使用情况的信息。

DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, 
show_counts=None)


verbose=None ：控制输出信息的详细程度。
buf=None ：一个打开的文件对象或类似文件的对象。如果提供，则输出将被写入这个缓冲区而不
是标准输出。
max_cols=None ：要显示的最大列数。如果DataFrame的列数超过这个值，则只会显示部分列的
信息。
memory_usage=None ：控制是否显示内存使用情况。可以设置为True或False，或者是一个字符
串，'deep’表示深度计算内存使用情况，考虑对象内部数据。
show_counts=None ：控制是否显示非空值的数量。如果设置为True，则会在每个列旁边显示非空
值的数量。

'''
import pandas as pd
# 创建一个DataFrame
df = pd.DataFrame({
 'A': [1, 2, 3, None],
 'B': ['a', 'b', 'c', 'd'],
 'C': [1.1, 2.2, 3.3, 4.4]
})
# 显示DataFrame的概要信息
df.info(verbose=True, memory_usage='deep', show_counts=True)


'''
apply
允许你对DataFrame中的每个元素、行或列应用一个函数。
DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)

func ：函数，应用于每个元素、行或列。这个函数需要你自定义，或者使用内置的函数。
axis=0 ：指定应用函数的轴。
raw=False ：布尔值，默认为False，则每行或每列将作为Series使用，如果为True，则将作为
Ndarray数组，性能会更好。
result_type=None ：控制返回类型。可以是以下之一：
  ‘reduce’：如果可能的话，如果应用函数返回一个列表，这个列表会被转换成一个 Series。
  ‘broadcast’：结果会被广播到原始 DataFrame 的形状，保留原始的索引和列。
  ‘expand’：如果应用函数返回一个列表（或类似列表的结构），则这个列表会被转换为多个
列。这意味着每个列表元素都会变成 DataFrame 的一列。
args=() ：元组，包含传递给函数的位置参数。
**kwargs ：关键字参数，将被传递给函数。
'''
import pandas as pd
# 创建一个DataFrame
df = pd.DataFrame({
 'A': [1, 2, 3],
 'B': [4, 5, 6],
 'C': [7, 8, 9]
})
# 定义一个函数，计算每行的平均值
def mean_row(row):
 return row.mean()
# 应用函数到每行
result = df.apply(mean_row, axis=1)
print(result)