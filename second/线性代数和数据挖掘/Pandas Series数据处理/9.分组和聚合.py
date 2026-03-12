# groupby
'''


用于将Series中的数据分组，并允许你对这些分组进行操作，比如计算每个组的总和、平均值、最大
值、最小值等。

Series.groupby(by=None, axis=0, level=None, as_index=True, sort=True,
group_keys=True, observed=False, dropna=True)

by ：确定分组依据，如果 by 是一个函数，它会在对象索引的每个值上调用。如果传递了字典或
Series，将使用这些对象的值来确定组。如果传递了长度等于所选轴的列表或 ndarray，则直接使
用这些值来确定组。
axis ：用于分组的轴。对于Series，这个参数通常设置为0（默认值），因为Series是一维数据结
构。
level ：如果索引是多级索引（MultiIndex），则此参数用于指定分组所依据的级别，by和level
同时只能存在一个，且必须存在一个。
as_index ：是否将分组键作为结果的索引，默认值为 True 。仅与DataFrame输入相关。
sort ：是否对结果进行排序。默认值为 True 。
group_keys ：是否在结果中包含分组键。默认值为 True 。
observed ：是否仅包含实际观察到的分类值。默认值为 False 。
dropna ：是否从结果中排除包含 NaN 的组。默认值为 True 。
'''
#原始Series对象的值分组
# import pandas as pd
# # 创建一个Series对象
# data = [10, 20, 10, 30, 20, 10]
# series = pd.Series(data)
# # 根据Series的值进行分组，并计算每组的计数
# grouped = series.groupby(series).count()
# print(grouped)



# import pandas as pd
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50, 60]
# index = ['a', 'b', 'a', 'b', 'c', 'c']
# series = pd.Series(data, index=index)
# # 根据Series的索引进行分组，并计算每组的均值
# grouped = series.groupby(series.index).sum()
# print(grouped)



# import pandas as pd
# import numpy as np
# arrays = [np.array(['qux', 'qux', 'foo', 'foo', 'baz', 'baz', 'bar', 'bar']),
#           np.array(['two', 'one', 'two', 'one','two', 'one', 'two', 'one'])]
# s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
# res = s.groupby(level=0).count()
# print(res)

















'''
agg 聚合操作
Series.agg(func=None,axis=0,*args,**kwargs)

'''

import pandas as pd

s = pd.Series([1,2,3,4,5])

result = s.agg('mean')
print(result)


result= s.agg(['max','min'])
print(result)


result = s.agg({'Maxinum':'max','Mininum':'min'})
print(result)



def custom_agg(x,power):
    return (x**power).sum()


result = s.agg(custom_agg,power=2)
print(result)