'''
数据清洗指对Series对象中的某些值进行删除、修改等操作，分别使用以下方法：
1. dropna()：删除包含NaN值的行
series.dropna(axis=0,inplace=False)


axis ：可选参数，用于指定按哪个轴删除缺失值。对于 Series 对象，因为它是一维数据结构，
只有一个轴，所以此参数默认值为 0 ，且一般不需要修改这个参数（在处理 DataFrame 时该参数
才有更多实际意义，如 axis = 0 表示按行删除， axis = 1 表示按列删除）。
inplace ：可选参数，用于指定是否在原 Series 对象上进行操作。如果 inplace = True ，则会
直接在原 Series 上删除缺失值，原 Series 会被修改；如果 inplace = False （默认值），则会
返回一个删除了缺失值的新 Series ，原 Series 保持不变。
'''
import pandas as pd
import numpy as np
# 创建一个包含缺失值的Series对象
data = [1, np.nan, 3, 4, np.nan]
series = pd.Series(data)
# 使用dropna函数删除缺失值，返回新的Series
new_series = series.dropna()
print(new_series)

'''
fillna()：填充NaN值。

Series.fillna(value=None, method=None, axis=None, inplace=False, limit=None, 
downcast=None)

value ：用于填充缺失值的标量值或字典。如果传递的是字典，则字典的键应该是要填充的标签，
值是用于填充的值。
method ：字符串，表示填充的方法。可选值包括：
pad / ffill ：用前一个非缺失值去填充缺失值。
bfill / backfill ：用后一个非缺失值去填充缺失值。
axis ：填充的轴，对于 Series 对象来说，这个参数通常不需要指定，因为 Series 是一维的。
inplace ：布尔值，表示是否在原地修改数据。如果为 True ，则直接在原 Series 上修改，不
返回新的对象。
limit ：整数，表示最大填充量。如果指定，则只填充前 limit 个缺失值。
downcast ：字典，用于向下转换数据类型。例如，可以将 float64 转换为 float32 。
'''

# import pandas as pd
# import numpy as np
# # 创建一个包含缺失值的 Series
# s = pd.Series([1, np.nan, 3, np.nan, 5])
# # 使用标量值填充
# filled_with_scalar = s.fillna(0)
# print(filled_with_scalar)
#
# #使用字典去填充
# filled_with_dict = s.fillna({'abc':2,3:'three'})#这里如果abc标签没有,则不会填充,仍然是Nan
# # 使用前向填充
# filled_with_ffill = s.fillna(method='ffill')
# print(filled_with_ffill)
# # 使用后向填充(用后一个非缺失值补充前一个)
# filled_with_bfill = s.fillna(method='bfill')
# print(filled_with_bfill)
# # 使用 limit 参数(限制填充次数)
# filled_with_limit = s.fillna(value=0, limit=1)
# print(filled_with_limit)



'''
isnull()：检测Series对象中的缺失值，它会返回一个布尔型Series，其中每个元素表示原Series对
应位置的值是否为缺失值（NaN）。


Series.isnull()
'''
# import pandas as pd
# import numpy as np
# # 创建一个包含缺失值的Series
# s = pd.Series([1, 2, np.nan, 4, np.nan])
# # 使用isnull()方法检测缺失值
# missing_values = s.isnull()
# print(missing_values)


'''
drop_duplicates()：用于去除Series对象中的重复项。

Series.drop_duplicates(keep='first', inplace=False, ignore_index=False)

keep：可选参数，决定了如何处理重复项。有三个选项：
'first' ：默认值，保留第一次出现的重复项。
'last' ：保留最后一次出现的重复项。
False ：不保留任何重复项，即删除所有重复项。
inplace ：布尔值，默认为 False 。如果设置为 True ，则直接在原始Series上进行操作，返回值
为 None 。如果设置为 False ，则返回一个新的Series，不修改原始Series。
ignore_index ：布尔值，默认为 False 。如果设置为 True ，则结果的索引将被重新设置，以反
映删除重复项后的新顺序。如果设置为 False ，则保留原始索引。
'''
import pandas as pd
series = pd.Series(['a', 'b', 'b', 'c', 'c', 'c', 'd'])
series_unique = series.drop_duplicates(keep='first')
print(series_unique)