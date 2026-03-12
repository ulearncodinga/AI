# 数据排序
# sort_values
'''
根据一个或者多个列的值对DataFrame进行排序

 DataFrame.sort_values(by, axis=0, ascending=True, inplace=False,
kind='quicksort', na_position='last', ignore_index=False, key=None)


by : 用于排序的列名或列名列表。
axis : {0 or ‘index’, 1 or ‘columns’}，默认为 0。沿着哪个轴进行排序。
ascending : 排序的方向，True表示升序，False表示降序， 默认为True。
inplace : 是否在原地修改 DataFrame 。
kind : {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}，默认为 ‘quicksort’。排序算法。
na_position : {‘first’, ‘last’}，默认为 ‘last’。缺失值的放置位置。
ignore_index : 布尔值，默认为 False 。是否忽略原来的索引，重新生成一个新的默认整数索
引。
key : 函数，默认为 None 。应用于 by 中每个列的函数，排序将基于函数的返回值。
'''

# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame({
#     'col1':['A','A','B',np.nan,'D','C'],
#     'col2':[2,1,9,8,7,4],
#     'col3':[0,1,9,4,2,3],
#     'clo4':['a','B','c','D','e','F']
#
# })
# print(df)
# result = df.sort_values(by=['col1'])
# print(result)
#
# # 根据 'col1' 和 'col2' 列对DataFrame进行排序
# res2 = df.sort_values(by=['col1', 'col2'])
# # 打印排序后的DataFrame
# print(res2)


#数据排序
# sort_index()：用于根据索引对 DataFrame 进行排序。
'''
DataFrame.sort_index(axis=0, level=None, ascending=True, inplace=False, 
kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, 
key=None)
axis : {0 or ‘index’, 1 or ‘columns’}，默认为 0。表示沿着哪个轴进行排序。0按照行标签排序，1
按照列标签排序。
level : 如果索引是多级索引，指定要排序的级别。可以是整数或整数列表。
ascending : 默认为 True 。表示排序是升序还是降序。
inplace : 是否在原地修改 DataFrame 。
kind : {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}，默认为 ‘quicksort’。排序算法。
na_position : {‘first’, ‘last’}，默认为 ‘last’。缺失值的放置位置。
sort_remaining : 是否对剩余的级别进行排序。仅在多级索引时有效。默认值为 True 。
ignore_index : 是否忽略原来的索引，重新生成一个新的默认整数索引。默认值为 False 。
key : 函数，默认为 None 。应用于索引的函数，排序将基于函数的返回值
'''
# import pandas as pd
# import numpy as np
# # 创建一个多级索引的DataFrame
# arrays = [np.array(['qux', 'qux', 'foo', 'foo']),
#     np.array(['two', 'one', 'two', 'one'])]
# df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]}, index=arrays)
# print(df)
# # 按第一层索引升序排序
# df_sorted_by_first_level = df.sort_index(level=0)
# print(df_sorted_by_first_level)
# # 按第二层索引降序排序
# df_sorted_by_second_level_desc = df.sort_index(level=1, ascending=False)
# print(df_sorted_by_second_level_desc)
# # 按整个索引升序排序
# df_sorted_by_full_index = df.sort_index(ascending=True)
# print(df_sorted_by_full_index)










# 数据筛选
# import pandas as pd
# data = {
#  '姓名': ['小明', '小红', '小刚'],
#  '年龄': [20, 18, 22],
#  '成绩': [85, 90, 88]
# }
# df = pd.DataFrame(data)
# # 使用布尔索引选择成绩大于或等于90的学生
# high_scores = df[df['成绩'] >= 90]
# print(high_scores)