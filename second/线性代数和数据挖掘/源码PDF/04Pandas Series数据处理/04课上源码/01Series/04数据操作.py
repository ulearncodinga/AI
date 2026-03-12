
# # dropna
# import pandas as pd
# import numpy as np
#
# # 创建一个包含缺失值的Series对象
# data = [1, np.nan, 3, 4, np.nan]
# series = pd.Series(data)
#
# print(series)
#
# # 使用dropna函数删除缺失值，返回新的Series
# new_series = series.dropna(inplace=True)
#
# print(new_series)
# print(series)



# # fillna
# import pandas as pd
# import numpy as np
#
# # 创建一个包含缺失值的 Series
# s = pd.Series([1, np.nan, 3, np.nan, 5])
#
# print(s)
#
# # # 使用标量值填充
# # filled_with_scalar = s.fillna(0)
# # print(filled_with_scalar)
# #
# #
# # filled_with_dict = s.fillna({'abc': 2, 3: 'three'})
# # print(filled_with_dict)
#
#
# # # 使用前向填充
# # filled_with_ffill = s.fillna(method='ffill')
# # print(filled_with_ffill)
#
#
# # # 使用后向填充
# # filled_with_bfill = s.fillna(method='bfill')
# # print(filled_with_bfill)
#
#
# # 使用 limit 参数
# filled_with_limit = s.fillna(value=0, limit=1)
# print(filled_with_limit)



# # isnull
# import pandas as pd
# import numpy as np
#
# # 创建一个包含缺失值的Series
# s = pd.Series([1, 2, np.nan, 4, np.nan], index=['a', 'b', 'c', 'd', 'e'])
#
# # 使用isnull()方法检测缺失值
# missing_values = s.isnull()
#
# print(missing_values)

#
# # drop_duplicates
# import pandas as pd
#
# series = pd.Series(['a', 'b', 'b', 'c', 'c', 'c', 'd'])
# print(series)
# series_unique = series.drop_duplicates(keep='last', ignore_index=True)
# print(series_unique)




# # replace
# import pandas as pd
#
# # 创建一个 Series
# s = pd.Series([1, 2, 3, 2, 5])
#
# # 进行替换操作
# replaced = s.replace(to_replace=2, value=20)
# print(replaced)
#

# # astype
# import pandas as pd
#
# s = pd.Series(['a', 'b', 'c'])
#
# print(s)
# # 将整数转换为字符串
# s_str = s.astype(int, errors='ignore')
# print(s_str)



# # transform
# import pandas as pd
#
#
# # 自定义一个函数，返回每个元素的平方
# def square(x, y):
#     print(y)
#     return x ** 2
#
#
# # 创建一个Series
# s = pd.Series([1, 2, 3, 4, 5])
#
# print(s)
# # 使用自定义函数对Series进行平方变换
# transformed_series = s.transform(square, y='hello')
#
# print(transformed_series)


# # sort_values
# import pandas as pd
# import numpy as np
#
#
# def square(x):
#     return x ** 2
#
# s = pd.Series([-3, 1, 4, 1, np.nan, 9], index=['a', 'b', 'c', 'd', 'e', 'f'])
#
# # 排序并重置索引
# sorted_s = s.sort_values(ascending=False, na_position='first', key=square)
# print(sorted_s)




# # # sort_index
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
# # print(s)
# # print(s.shape)
#
# res = s.sort_index(level=2, ascending=True, sort_remaining=True, ignore_index=False)
# print(res)


# # 筛选
# import pandas as pd
#
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
#
# print(series)
#
# # 选择大于30的元素
# print(series[series > 40])



import pandas as pd

# 创建三个Series
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'f'])
s3 = pd.Series([7, 8, 9], index=['e', 'f', 'g'])

# 使用concat连接Series
result = pd.concat([s1, s2, s3])

print(result)