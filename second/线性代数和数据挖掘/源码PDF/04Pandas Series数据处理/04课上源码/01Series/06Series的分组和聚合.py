


# # groupby 分组
# import pandas as pd
#
# # 创建一个Series对象
# data = [20, 20, 10, 30, 20, 10]
# series = pd.Series(data)
#
# # 根据Series的值进行分组，并计算每组的计数
# grouped = series.groupby(series, sort=True).count()
#
# print(grouped)

#
# import pandas as pd
#
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50, 60]
# index = ['a', 'b', 'a', 'b', 'c', 'c']
# series = pd.Series(data, index=index)
#
# # 根据Series的索引进行分组，并计算每组的均值
# grouped = series.groupby(series.index).sum()
#
# print(grouped)


#
# import pandas as pd
# import numpy as np
#
# arrays = [np.array(['qux', 'qux', 'foo', 'foo',
#                     'baz', 'baz', 'bar', 'bar']),
#           np.array(['two', 'one', 'two', 'one',
#                     'two', 'one', 'two', 'one'])]
#
#
# s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
# print(s)
# res = s.groupby(level=0, by=s).sum()
# print(res)



# agg 聚合
import pandas as pd

s = pd.Series([1, 2, 3, 4, 5])

# # 计算平均值
# result = s.agg('min')
# print(result)

# # 计算最大值和最小值
# result = s.agg(['max', 'min', 'mean'])
# print(result)

# # 使用字典为聚合函数命名
# result = s.agg({'Maximum': 'max', 'Minimum': 'min'})
# print(result)


# 使用自定义函数并传递额外的参数
def custom_agg(x, power):
    return (x ** power).sum()

result = s.agg(custom_agg, power=2)
print(result)