# isnull:检测缺失值
# dropna()：用于删除 DataFrame 中的缺失值。
'''DataFrame.dropna(axis=0, how=any, thresh=_NoDefault.no_default,
subset=None, inplace=False, ignore_index=False)
'''
# import pandas as pd
# import numpy as np
# # 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#  'A': [1, 2, np.nan],
#  'B': [4, np.nan, 6],
#  'C': [7, 8, 9]
# })
# # 打印原始DataFrame
# print(df)
# # 删除任何含有 NaN 值的行
# df_cleaned = df.dropna()
# print(df_cleaned)



#fillna()：用于填充 DataFrame 中的缺失值。
'''
DataFrame.fillna(value=None, *, method=None, axis=0, 
inplace=False, limit=None)

value : 填充值，可以是单个值，也可以是字典（对不同的列填充不同的值），或
者一个 Series。
method : {‘bfill’, ‘ffill’}，默认为无默认值。指定填充方法：
‘bfill’ 或 ‘backfill’：使用下一个有效观测值填充。
‘ffill’ 或 ‘pad’：使用前一个有效观测值填充。
axis : {0 或 ‘index’, 1 或 ‘columns’}，默认为0。
inplace : 布尔值，默认为 False。如果为 True，则在原地修改 DataFrame 而不
返回新的 DataFrame。
limit : int，默认为无默认值。如果指定了 method ，则该参数限制连续填充的
数量。
'''
#
# import pandas as pd
# import numpy as np
# # 创建一个包含缺失值的 DataFrame
# df = pd.DataFrame({
#   'A': [1, 2, np.nan],
#   'B': [np.nan, np.nan, 6],
#   'C': [7, np.nan, 9]
# })
# # 打印原始DataFrame
# print(df)
# # 使用固定值填充缺失值
# df_filled_value = df.fillna(value=0)
# print(df_filled_value)
# # 使用字典填充
# data = {
#    'A': 'a',
#    'B': 'b',
#    'C': 'c'
# }
# df_filled_dict = df.fillna(value=data)
# print(df_filled_dict)
# # 使用Series填充
# data_series = pd.Series(['a', 'b', 'c'], ['A', 'B', 'C'])
# df_filled_series = df.fillna(value=data_series)
# print(df_filled_series)
# # 使用前一个有效观测值填充缺失值
# df_filled_ffill = df.ffill( )
# print(df_filled_ffill)
# # 使用后一个有效观测值填充缺失值
# df_filled_bfill = df.bfill( )
# print(df_filled_bfill)

'''
drop_duplicates()：用于删除 DataFrame 中的重复行。

DataFrame.drop_duplicates(subset=None, keep='first', 
inplace=False, ignore_index=False)
'''
import pandas as pd
# 创建一个包含重复行的 DataFrame
df = pd.DataFrame({
 'A': [1, 1, 2, 2, 3, 3],
 'B': [1, 1, 2, 2, 3, 3],
 'C': [1, 1, 2, 2, 3, 3]
})
# 打印原始DataFrame
print(df)
# 删除重复行，保留第一次出现的重复项
df_dedup_first = df.drop_duplicates()
print(df_dedup_first)
# 根据指定列删除重复行
df_dedup_column = df.drop_duplicates(subset=['A'])
print(df_dedup_column)
# 删除重复行，保留最后一次出现的重复项
df_dedup_last = df.drop_duplicates(keep='last')
print(df_dedup_last)
# 删除所有重复行
df_dedup_all = df.drop_duplicates(keep=False)
print(df_dedup_all)