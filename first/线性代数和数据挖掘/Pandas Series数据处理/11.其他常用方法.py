'''
1. unique
'''
# import pandas as pd
# # 创建一个 Series
# s = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])
# # 获取唯一值
# unique_values = s.unique()
# # 输出唯一值
# print(unique_values)

'''
2.  nunique
该函数用于计算 Series 中唯一值的数量。这个方法返回一个整数，表示 Series 中唯一值的数量
# '''
# import pandas as pd
# # 创建一个 Series
# s = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])
# # 计算唯一值的数量（排除 NaN）
# unique_count = s.nunique(dropna=True)
# # 输出唯一值的数量
# print(unique_count)

'''
3. value_counts
该方法用于计算 Series 中每个值的出现次数。这个方法返回一个包含每个唯一值及其对应出现次数的
Series。


Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, 
dropna=True)

normalize: 布尔值或 ‘all’，默认为 False 。如果为 True ，返回每个值的相对频率；如果为 ‘all’，
则返回所有值的相对频率之和为 1。
sort: 布尔值，默认为 True 。如果为 True ，结果将按计数值降序排序。
ascending: 布尔值，默认为 False 。如果为 True ，结果将按计数值升序排序。
bins: 用于离散化连续数据，可以是整数或分位数数组。如果指定了 bins ，则结果将是每个 bin 
的计数。
dropna: 布尔值，默认为 True 。如果为 True ，则排除 NaN 值。
'''
# import pandas as pd
# # 创建一个 Series
# s = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])
# # 计算每个值的出现次数
# value_counts = s.value_counts()
# # 输出每个值的出现次数
# print(value_counts)



'''
4 describe
该方法用于生成描述性统计信息。这个方法返回一个包含计数、均值、标准差、最小值、25% 分位数、
中位数、75% 分位数和最大值的 Series

Series.describe(percentiles=None, include=None, exclude=None)

percentiles: 数值列表或数值元组，默认为 [.25, .5, .75] ，表示要包含在输出中的分位数。
include: 字符串或类型列表，用于指定要包括在结果中的数据类型。默认为 None ，即包括所有数
字类型。
exclude: 字符串或类型列表，用于指定要从结果中排除的数据类型。默认为 None 。
'''
# import pandas as pd
# # 创建一个 Series
# s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # 生成描述性统计信息
# description = s.describe()
# # 输出描述性统计信息
# print(description)

'''
5 copy
该函数用于创建 Series 对象的一个副本。
Series.copy(deep=True)
'''
# import pandas as pd
# # 创建一个 Series
# original_series = pd.Series([1, 2, 3, 4, 5])
# # 创建一个深拷贝
# deep_copied_series = original_series.copy()
# # 创建一个浅拷贝
# shallow_copied_series = original_series.copy(deep=False)
# # 修改深拷贝中的数据
# deep_copied_series[0] = 999
# # 输出原始 Series 和深拷贝后的 Series
# print("Original Series:\n", original_series)
# print("Deep Copied Series:\n", deep_copied_series)
# # 修改浅拷贝中的数据
# shallow_copied_series[1] = 888
# # 输出原始 Series 和浅拷贝后的 Series
# print("Original Series after shallow copy modification:\n", original_series)
# print("Shallow Copied Series:\n", shallow_copied_series)


'''
6 reset_index
该函数用于重置 Series 的索引，将原来的索引转换为一个列，并将一个新的默认整数索引赋给 Series。

Series.reset_index(level=None, *, drop=False, name=no_default, inplace=False, 
allow_duplicates=False)
'''
# import numpy as np
# import pandas as pd
# data = pd.Series(np.random.randint(1, 100, 5), index=['c', 'a', 'e', 'b', 'd'])
# # 对Series进行排序
# sorted_series = data.sort_values()
# # 重置索引
# reset_indexed_series = sorted_series.reset_index(drop=True)
# print("排序后的Series：")
# print(sorted_series)
# print("重置索引后的Series：")
# print(reset_indexed_series)



'''
7 info
用于显示Series的概要信息的方法。这个方法提供了关于Series的元数据，包括数据类型、非空值的数
量、内存使用情况等。

Series.info(verbose=None, buf=None, max_cols=None, memory_usage=None, 
show_counts=True)

'''
# import numpy as np
# import pandas as pd
# # 创建一个示例Series
# s = pd.Series([1, 2, np.nan, 4, 5], name='example_series', index=['a', 'b', 'c',
# 'd', 'e'])
# # 显示Series的概要信息
# s.info()


'''
.8 apply
对 Series 中的每个元素应用一个函数，并返回一个结果 Series 。
Series.apply(func, convert_dtype=True, args=(), **kwargs)
'''

# import pandas as pd
# # 创建一个Series
# series = pd.Series([1, 2, 3, 4, 5])
# # 使用 apply 方法结合 lambda 函数，对 series 中的每个元素执行平方操作
# res = series.apply(lambda x: x ** 2)
# # 打印结果，输出每个元素的平方值
# print(res)


'''
9 map
对 Series 中的每个元素应用一个映射，它允许你将一个函数应用到 Series 的每个元素上，或者将一
个字典或 Series 映射到 Series 的值上。

Series.map(arg, na_action=None)
'''
# import pandas as pd
# # 创建一个 Series
# s = pd.Series([1, 2, 3, 4, 5])
# # 定义一个函数，用于将值翻倍
# def double(x):
#     return x * 2
# # 使用 map 方法应用这个函数
# s_doubled = s.map(double)
# print(s_doubled)



# import pandas as pd
# # 创建一个映射字典
# grade_mapping = {
#  90: 'A',
#  80: 'B',
#  70: 'C',
#  60: 'D',
#  100: 'F'
# }
# # 创建一个成绩的 Series
# grades = pd.Series([80, 92, 77, 59, 100])
# # 使用 map 方法应用这个字典
# grades_mapped = grades.map(grade_mapping)
# print(grades_mapped)



import pandas as pd
# 创建一个名为series1的Series对象
series1 = pd.Series([50, 60, 70, 80, 90], index=['a', 'b', 'c', 'd', 'e'])
# 创建一个名为grades的Series对象，代表成绩数据
grades = pd.Series([80, 92, 77, 59, 100], index=[0, 1, 2, 3, 4])
# 使用grades的map方法，将grades中的每个值作为键，去series1中查找对应的键值并返回
res = grades.map(series1)
# 打印输出新生成的Series对象res
print(res)