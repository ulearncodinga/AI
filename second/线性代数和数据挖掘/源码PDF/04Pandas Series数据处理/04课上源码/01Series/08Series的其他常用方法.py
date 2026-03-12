

# # unique
# import pandas as pd
#
# # 创建一个 Series
# s = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])
#
# # 获取唯一值
# unique_values = s.unique()
#
# # 输出唯一值
# print(unique_values)
# print(type(unique_values))
# print(len(unique_values))
# num = s.nunique()
# print(num)


# # value_counts
# import pandas as pd
#
# # 创建一个 Series
# s = pd.Series(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])
#
# # 计算每个值的出现次数
# value_counts = s.value_counts(ascending=True)
#
# # 输出每个值的出现次数
# print(value_counts)


#
# import pandas as pd
#
# # 创建一个 Series
# s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#
# # 生成描述性统计信息
# description = s.describe()
#
# # 输出描述性统计信息
# print(description)



#
# import pandas as pd
#
# # 创建一个 Series
# original_series = pd.Series([1, 2, 3, 4, 5])
#
# # 创建一个深拷贝
# deep_copied_series = original_series.copy()
#
# # 创建一个浅拷贝
# shallow_copied_series = original_series.copy(deep=False)
#
# # 修改深拷贝中的数据
# deep_copied_series[0] = 999
#
# # # 输出原始 Series 和深拷贝后的 Series
# # print("Original Series:\n", original_series)
# # print("Deep Copied Series:\n", deep_copied_series)
#
# # 修改浅拷贝中的数据
# shallow_copied_series[1] = 888
#
# # 输出原始 Series 和浅拷贝后的 Series
# print("Original Series after shallow copy modification:\n", original_series)
# print("Shallow Copied Series:\n", shallow_copied_series)



#
# import pandas as pd
# import numpy as np
#
# # 创建一个简单的Series对象
# data = pd.Series(np.random.randint(1, 100, 5), index=['c', 'a', 'e', 'b', 'd'])
#
#
# # 对Series进行排序
# sorted_series = data.sort_values()
#
# # 重置索引
# reset_indexed_series = sorted_series.reset_index(drop=False)
# print("排序后的Series：")
# print(sorted_series)
# print("重置索引后的Series：")
# print(reset_indexed_series)



#
# import numpy as np
# import pandas as pd
#
# # 创建一个示例Series
# s = pd.Series([1, 2, np.nan, 4, 5], name='example_series', index=['a', 'b', 'c', 'd', 'e'])
#
#
# # 显示Series的概要信息
# print(s.info())




#
# import pandas as pd
#
# # 创建一个Series
# series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
#
# # 使用 apply 方法结合 lambda 函数，对 series 中的每个元素执行平方操作
# res = series.apply(lambda x: x ** 2)
#
# # 打印结果，输出每个元素的平方值
# print(res)




# import pandas as pd
#
# # 创建一个 Series
# s = pd.Series([1, 2, 3, 4, 5])
#
# # 定义一个函数，用于将值翻倍
# def double(x):
#     return x * 2
#
# # 使用 map 方法应用这个函数
# s_doubled = s.map(double)
# print(s_doubled)
#

#
# import pandas as pd
#
# # 创建一个映射字典
# grade_mapping = {
#     90: 'A',
#     80: 'B',
#     70: 'C',
#     60: 'D',
#     0: 'F'
# }
#
# # 创建一个成绩的 Series
# grades = pd.Series([80, 92, 77, 80, 100])
#
# # 使用 map 方法应用这个字典
# grades_mapped = grades.map(grade_mapping)
# print(grades_mapped)



import pandas as pd

# 创建一个名为series1的Series对象
series1 = pd.Series([50, 60, 70, 80, 90], index=['a', 'b', 80, 'd', 'e'])

# 创建一个名为grades的Series对象，代表成绩数据
grades = pd.Series([80, 92, 77, 59, 100], index=[0, 1, 2, 3, 4])

print(series1)
print(grades)

# 使用grades的map方法，将grades中的每个值作为键，去series1中查找对应的键值并返回
res = grades.map(series1)

# 打印输出新生成的Series对象res
print(res)