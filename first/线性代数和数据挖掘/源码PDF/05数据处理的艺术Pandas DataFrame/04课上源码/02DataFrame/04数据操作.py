

# 数据清洗
# import pandas as pd
# import numpy as np
#
# # 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan],
#     'B': [4, np.nan, 6],
#     'C': [7, 8, 9]
# })
#
# # 打印原始DataFrame
# print(df)
#
# # 使用 isnull() 方法检测缺失值
# missing_values = df.isnull()
#
# print(missing_values)



# import pandas as pd
# import numpy as np
#
# # 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, np.nan, np.nan],
#     'B': [4, np.nan, 6],
#     'C': [7, 8, 9]
# })
#
# # 打印原始DataFrame
# print(df)
#
# # 删除任何含有 NaN 值的行
# df_cleaned = df.dropna(subset=['B'])
#
# print(df_cleaned)



# import pandas as pd
# import numpy as np
#
# # 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, np.nan],
#     'B': [np.nan, np.nan, 6],
#     'C': [7, np.nan, 9]
# })
#
# # 打印原始DataFrame
# print(df)
#
# # # 使用固定值填充缺失值
# # df_filled_value = df.fillna(value=0, limit=1, axis=1)
# # print(df_filled_value)
#
# # # 使用字典填充
# # data = {
# #     'A': 'a',
# #     'B': 'b',
# #     'C': 'c'
# # }
# # df_filled_dict = df.fillna(value=data)
# # print(df_filled_dict)
#
# # # 使用Series填充
# # data_series = pd.Series(['a', 'b', 'c'], ['A', 'B', 'C'])
# # df_filled_series = df.fillna(value=data_series)
# # print(df_filled_series)
#
# # 使用前一个有效观测值填充缺失值
# df_filled_ffill = df.fillna(method='ffill')
# print(df_filled_ffill)
#
# # 使用后一个有效观测值填充缺失值
# df_filled_bfill = df.fillna(method='bfill', axis=1)
# print(df_filled_bfill)



#
# import pandas as pd
#
# # 创建一个包含重复行的 DataFrame
# df = pd.DataFrame({
#     'A': [1, 1, 2, 2, 3, 3],
#     'B': [1, 1, 2, 3, 3, 3],
#     'C': [1, 1, 2, 2, 3, 3]
# })
#
# # 打印原始DataFrame
# print(df)
#
# # # 删除重复行，保留第一次出现的重复项
# # df_dedup_first = df.drop_duplicates(keep=False)
# # print(df_dedup_first)
#
# # # 根据指定列删除重复行
# # df_dedup_column = df.drop_duplicates(subset=['A'])
# # print(df_dedup_column)
#
# # # 删除重复行，保留最后一次出现的重复项
# # df_dedup_last = df.drop_duplicates(keep='last')
# # print(df_dedup_last)
# #
# # # 删除所有重复行
# # df_dedup_all = df.drop_duplicates(keep=False)
# # print(df_dedup_all)



#
# import pandas as pd
#
# # 创建一个 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 1, 4, 5],
#     'B': ['a', 'b', 'a', 'b', 'a']
# })
#
# # 打印原始DataFrame
# print(df)
#
#
# # data = {
# #     1: 20
# # }
# #
# # # 用数字 100 替换所有的 1
# # df_replaced = df.replace(to_replace=data)
#
# # # 用字符串 'z' 替换所有的 'a'
# # df_replaced = df.replace(to_replace='a', value='z')
#
# # # 使用字典替换多个值
# # df_replaced = df.replace({
# #     2: 200,
# #     'b': 'y'
# # })
#
# # 使用正则表达式替换
# df_replaced = df.replace(to_replace=r'^a$', value='z', regex=True)
#
# print(df_replaced)



# import pandas as pd
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
#     'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
#     'baz': [1, 2, 3, 4, 5, 6],
#     'zoo': ['x', 'y', 'z', 'q', 'w', 't']
# })
#
# # 打印原始DataFrame
# print(df)
#
# # # 使用pivot方法对DataFrame进行重塑，其中foo作为行索引，bar作为列索引，baz作为值
# # res1 = df.pivot(index='foo', columns='bar', values='baz')
# #
# # # 打印重塑后的DataFrame
# # print(res1)
#
# # 使用pivot方法对DataFrame进行重塑，其中foo作为行索引，bar作为列索引，baz、zoo作为值
# res2 = df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
# # 打印重塑后的DataFrame
# print(res2)


#
# import pandas as pd
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     'A': {0: 'a', 1: 'b', 2: 'c'},
#     'B': {0: 1, 1: 3, 2: 5},
#     'C': {0: 2, 1: 4, 2: 6}
# })
#
# # 打印原始DataFrame
# print(df)
#
# # 使用melt方法对DataFrame进行重塑
# res1 = df.melt(id_vars=['A'], value_vars=['B'],)
#
# # 打印重塑后的DataFrame
# print(res1)



# import numpy as np
# import pandas as pd
#
# # 创建一个DataFrame
# df = pd.DataFrame({
#     "A": ["foo", "foo", "foo", "foo", "foo",
#           "bar", "bar", "bar", "bar"],
#     "B": ["one", "one", "one", "two", "two",
#           "one", "one", "two", "two"],
#     "C": ["small", "large", "large", "small",
#           "small", "large", "small", "small",
#           "large"],
#     "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
#     "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]
# })
#
# # 打印原始DataFrame
# print(df)
#
# # 使用pivot_table方法创建一个数据透视表
# table = df.pivot_table(values=['D', 'E'], index=['A',], columns=['C'], aggfunc=np.sum, fill_value='a', margins=True, margins_name='test')
#
# # 打印数据透视表
# print(table)


#
# import pandas as pd
#
# # 创建一个示例 DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4.5, 5.5, 6.5],
#     'C': ['7', '8', '9']
# })
#
# # 打印原始DataFrame
# print(df)
#
# # 将列 'A' 转换为浮点数类型
# df['A'] = df['A'].astype(float)
#
# # 使用字典将多列转换为不同的数据类型
# # 将列 'B' 转换为整数类型，列 'C' 也转换为整数类型
# df = df.astype({
#     'B': int,
#     'C': int
# })
#
# # 打印转换后的DataFrame
# print(df)
#
# # 打印DataFrame中各列的数据类型
# print(df.dtypes)


#
# import numpy as np
# import pandas as pd
#
# # 创建一个示例 DataFrame
# df = pd.DataFrame({
#     'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
#     'col2': [2, 1, 9, 8, 7, 4],
#     'col3': [3, 1, 9, 4, 2, 3],
#     'col4': ['a', 'B', 'c', 'D', 'e', 'F']
# })
#
# # 打印原始DataFrame
# print(df)
#
# # # 根据 'col1' 列对DataFrame进行排序
# # res1 = df.sort_values(by=['col1'])
# # # 打印排序后的DataFrame
# # print(res1)
#
# # 根据 'col1' 和 'col2' 列对DataFrame进行排序
# res2 = df.sort_values(by=['col1', 'col3'])
# # 打印排序后的DataFrame
# print(res2)



# import pandas as pd
# import numpy as np
#
# # 创建一个多级索引的DataFrame
# arrays = [np.array(['qux', 'qux', 'foo', 'foo']),
#           np.array(['two', 'one', 'two', 'one'])]
# df = pd.DataFrame({'C': [1, 2, 3, 4], 'B': [4, 3, 2, 1]}, index=arrays)
#
# print(df)
#
# # # 按第一层索引升序排序
# # df_sorted_by_first_level = df.sort_index(level=0)
# # print(df_sorted_by_first_level)
#
# # # 按第二层索引降序排序
# # df_sorted_by_second_level_desc = df.sort_index(level=1, ascending=False)
# # print(df_sorted_by_second_level_desc)
#
# # 按整个索引升序排序
# df_sorted_by_full_index = df.sort_index()
# print(df_sorted_by_full_index)


#
# import pandas as pd
#
# data = {
#     '姓名': ['小明', '小红', '小刚'],
#     '年龄': [20, 18, 22],
#     '成绩': [85, 90, 88]
# }
#
# df = pd.DataFrame(data)
#
# print(df)
#
# print(df['成绩'] >= 90)
#
# # 使用布尔索引选择成绩大于或等于90的学生
# high_scores = df[df['成绩'] >= 90]
#
# print(high_scores)


#
# import pandas as pd
#
# # 创建两个 DataFrame
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                     'B': ['B0', 'B1', 'B2', 'B3'],
#                     'C': ['C0', 'C1', 'C2', 'C3'],
#                     'D': ['D0', 'D1', 'D2', 'D3']},
#                    index=[0, 1, 2, 3])
#
# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                     'B': ['B4', 'B5', 'B6', 'B7'],
#                     'C': ['C4', 'C5', 'C6', 'C7'],
#                     'F': ['F4', 'F5', 'F6', 'F7']},
#                    index=[4, 5, 6, 7])
#
# print(df1)
# print(df2)
#
# # 沿着竖直方向拼接两个DataFrame
# result = pd.concat([df1, df2], axis=1, join='outer')
#
# print(result)





# import pandas as pd
#
# # 创建两个 DataFrame
# df1 = pd.DataFrame({'key1': ['A', 'B', 'C', 'D'],
#                     'value': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
# df2 = pd.DataFrame({'key2': ['B', 'D', 'D', 'E'],
#                     'value': [5, 6, 7, 8]}, index=['a', 'c', 'e', 'f'])
#
# print(df1)
# print(df2)
#
# # 使用内连接（inner join）合并两个 DataFrame
# result = df1.merge(df2, left_on='key1', right_on='key2', how='right', suffixes=('_left', '_right'), indicator=True,)
#
# print(result)


import pandas as pd

# 创建两个 DataFrame
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': [1, 2, 3, 4]},
                   index=['k0', 'k1', 'k2', 'k3'])
df2 = pd.DataFrame({'value': [5, 6, 7, 8]},
                   index=['k1', 'k2', 'k3', 'k4'])

print(df1)
print(df2)

# 使用左连接（left join）根据索引合并两个 DataFrame
result = df1.join(df2, how='left', rsuffix='_right', lsuffix='_left')

print(result)