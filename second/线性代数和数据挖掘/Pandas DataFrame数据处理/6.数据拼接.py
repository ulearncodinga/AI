# 1. concat()：用于沿一个轴将多个 pandas 对象连接在一起。


'''
pandas.concat(objs, axis=0, join='outer', ignore_index=False, keys=None,
levels=None, names=None, verify_integrity=False, sort=False, copy=True)

objs : 要连接的对象列表。
axis : {0, 1, ‘index’, ‘columns’}，默认为 0。
join : 连接方式。可以是：
'outer' ：取所有索引的并集。
'inner' ：取所有索引的交集。
ignore_index : 是否忽略原来的索引，重新生成一个新的默认整数索引。默认值为 False 。
keys : 用于生成多级索引的键列表。每个键对应一个对象。
levels : 用于多级索引的级别列表。通常与 keys 一起使用。
names : 用于多级索引的名称列表。通常与 keys 一起使用。
verify_integrity : 是否验证最终的 DataFrame 是否有重复的索引。默认值为 False 。
sort : 是否对结果按照列名进行升序排序。默认值为 False 。
copy : 是否复制数据。
'''
# import pandas as pd
# # 创建两个 DataFrame
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#   'B': ['B0', 'B1', 'B2', 'B3'],
#  'C': ['C0', 'C1', 'C2', 'C3'],
#  'D': ['D0', 'D1', 'D2', 'D3']},
#   index=[0, 1, 2, 3])
# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#  'B': ['B4', 'B5', 'B6', 'B7'],
#  'C': ['C4', 'C5', 'C6', 'C7'],
#  'F': ['F4', 'F5', 'F6', 'F7']},
#   index=[4, 5, 6, 7])
# # 沿着竖直方向拼接两个DataFrame
# result = pd.concat([df1, df2], axis=0, ignore_index=True)
# print(result)



# 2. merge()：用于根据一个或多个键将两个 DataFrame 对象连接起来
'''
DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, 
left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), 
copy=None, indicator=False, validate=None)

right : 另一个 DataFrame 对象。
how : {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, 默认为 ‘inner’。确定连接的类型：
‘left’: 使用左侧（调用 merge的 DataFrame）的索引进行左连接。
‘right’: 使用右侧（参数 right 中的 DataFrame）的索引进行右连接。
‘outer’: 使用两个DataFrame的并集连接。
‘inner’: 使用两个DataFrame的交集连接。
on : 用于合并的列名。如果 left_on 和 right_on 都没有指定，则使用 on 。
left_on : 左侧 DataFrame 中用于合并的列名。不与on同时使用。
right_on : 右侧 DataFrame 中用于合并的列名。不与on同时使用。
left_index : 是否使用左侧 DataFrame 的索引作为合并键。默认值为 False 。不与on同时使
用。
right_index : 是否使用右侧 DataFrame 的索引作为合并键。默认值为 False 。不与on同时使
用。
sort : 是否对结果进行排序。默认值为 False 。
suffixes : 用于重命名重复列的后缀。默认值为 ('_x', '_y') 。
copy : 是否复制数据。默认值为 None ，表示根据需要自动决定是否复制。
indicator : 是否添加一个指示器列，显示每行来自哪个 DataFrame 。默认值为 False 。
validate : 检查合并键。可以是：
'one_to_one' ：检查合并键在两者中是否唯一。
'one_to_many' ：检查合并键在左侧是否唯一。
'many_to_one' ：检查合并键在右侧是否唯一。
'many_to_many' ：不检查v
'''
# import pandas as pd
# # 创建两个 DataFrame
# df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
# 'value': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
# df2 = pd.DataFrame({'key': ['B', 'D', 'D', 'E'],
# 'value': [5, 6, 7, 8]}, index=['a', 'c', 'e', 'f'])
# # 使用内连接（inner join）合并两个 DataFrame
# result = df1.merge(df2, on='key', how='inner', suffixes=('_left', '_right'))
# print(result)




# 3. join()：用于将两个对象的列连接起来。
'''
DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False, 
validate=None)
other : 另一个 DataFrame 对象。
on : 用于连接的列名
how : {‘left’, ‘right’, ‘outer’, ‘inner’}, 默认为 ‘left’。确定连接的类型：
‘left’: 使用左侧（调用 join 的 DataFrame）的索引进行左连接。
‘right’: 使用右侧（参数 other 中的 DataFrame）的索引进行右连接。
‘outer’: 使用两个 DataFrame 的索引的并集进行全外连接。
‘inner’: 使用两个 DataFrame 的索引的交集进行内连接。
lsuffix : 用于重命名重复列的左后缀。默认值为空字符串 '' 。
rsuffix : 用于重命名重复列的右后缀。默认值为空字符串 '' 。
sort : 是否对结果进行排序。默认值为 False 。
validate : 检查合并键。可以是：
'one_to_one' ：检查合并键在两者中是否唯一。
'one_to_many' ：检查合并键在左侧是否唯一。
'many_to_one' ：检查合并键在右侧是否唯一。
'many_to_many' ：不检查。
'''
import pandas as pd
# 创建两个 DataFrame
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
'value': [1, 2, 3, 4]},
index=['k0', 'k1', 'k2', 'k3'])
df2 = pd.DataFrame({'value2': [5, 6, 7, 8]},
index=['k1', 'k2', 'k3', 'k4'])
# 使用左连接（left join）根据索引合并两个 DataFrame
result = df1.join(df2, how='left')
print(result)