'''
1. sort_values()：按照值对 Series 进行排序。


Series.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort',
na_position='last', ignore_index=False, key=None)

axis：默认为 0。对于 Series，这个参数不起作用，因为 Series 是一维的，而 sort_values 总是
在 axis=0 上操作。
ascending：布尔值，默认为 True 。如果是 True ，则按照升序排列；如果是 False ，则按照降
序排列。
inplace：布尔值，默认为 False 。如果为 True ，则排序将直接在原始 Series 上进行，不返回新
的 Series。
kind：排序算法，{‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}，默认为 ‘quicksort’。决定了使用
的排序算法。
na_position：{‘first’, ‘last’}，默认为 ‘last’。这决定了 NaN 值的放置位置。
ignore_index：布尔值，默认为 False 。如果为 True ，则排序后的 Series 将重置索引，使其成
为默认的整数索引。
key：函数，默认为 None 。如果指定，则这个函数将在排序之前应用于每个值，并且排序将基于
这些函数返回的值。

'''
# import pandas as pd
# import numpy as np
# def square(x):
#     return x ** 2
# s = pd.Series([-3, 1, 4, 1, np.nan, 9], index=['a', 'b', 'c', 'd', 'e', 'f'])
# # 排序并重置索引
# sorted_s = s.sort_values(ignore_index=True, key=square)
# print(sorted_s)

'''
2. sort_index()：按照索引的顺序对数据进行排序。

Series.sort_index(axis=0, level=None, ascending=True, inplace=False, 
kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, 
key=None)

axis : 默认为0，对于 Series，这个参数不起作用。
level : 默认为 None，如果索引是多级索引（也称为层次化索引或 MultiIndex），则可以指定要排
序的级别。
ascending : 默认为 True，如果为 True，则按升序排序；如果为 False，则按降序排序。对于多级
索引，可以传递一个布尔值列表，以指定每个级别的排序顺序。
inplace : 默认为 False，如果为 True，则直接在原对象上进行修改，不会返回一个新的对象。
kind : {‘quicksort’, ‘mergesort’, ‘heapsort’}，默认为 ‘quicksort’，指定排序算法。‘quicksort’ 是最
快的通用排序算法，但不是稳定的；‘mergesort’ 是稳定的，但可能比 ‘quicksort’ 慢；‘heapsort’ 
是原地排序算法，但通常比其他两个选项慢。
na_position : {‘first’, ‘last’}, 默认为 ‘last’，指定 NaN 值应该排在排序结果的开头还是结尾。
sort_remaining : 默认为 True，对于多级索引，如果为 True，在该level排序后，在排序的基础上
对剩下的级别的元素还会排序。
ignore_index : 默认为 False，如果为 True，则排序后的结果将不再保留原始索引，而是使用默认
的整数索引。
key：函数，默认为 None 。如果指定，则这个函数将在排序之前应用于每个值，并且排序将基于
这些函数返回的值。
'''
# import pandas as pd
# import numpy as np
# arrays = [np.array(['qux', 'qux', 'foo', 'foo','baz', 'baz', 'bar', 'bar']),
#  np.array(['two', 'one', 'two', 'one','two', 'one', 'two', 'one'])]
# s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=arrays)
# print(s)
# res = s.sort_index(level=1, ascending=True, sort_remaining=True,
# ignore_index=False)
# print(res)


'''
数据筛选
可以使用一个布尔数组来选择满足条件的元素
'''
# import pandas as pd
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
# # 选择大于30的元素
# print(series[series > 30])



'''
数据拼接

concat()：用于将多个Pandas对象（如Series或DataFrame）沿着一个轴连接起来的函数。


pandas.concat(objs, *, axis=0, join='outer', ignore_index=False, keys=None, 
levels=None, names=None, verify_integrity=False, sort=False, copy=None)



objs : 参与连接的Pandas对象的列表或元组。例如，可以是多个Series或DataFrame。
axis : {0或’index’, 1或’columns’}，表示连接的轴，默认为0。0表示沿着行方向连接（索引轴），
1表示沿着列方向连接（列轴）。
join : {‘inner’, ‘outer’}，默认为’outer’。如何处理其他轴上的索引。'outer’表示并集，保留所有索
引；'inner’表示交集，只保留所有对象共有的索引。
ignore_index : 布尔值，默认为False。如果为True，则不保留原索引，而是创建一个新索引，可
以避免重复的索引。
keys : 序列，默认为None。用于创建分层索引的键。如果提供了keys，则生成的DataFrame或
Series将具有分层索引。
levels : 序列列表，默认为None。用于构造分层索引的特定级别，如果设置了keys，则默认为
keys。
names : 列表，默认为None。生成的分层索引中的级别名称。如果提供了keys，表示使用keys作
为索引名称。
verify_integrity : 布尔值，默认为False。如果为True，则检查新连接的轴是否包含重复的索
引，如果发现重复，则引发ValueError。这在确保数据没有重复时很有用。
sort : 布尔值，默认为False。在连接之前是否对非连接轴上的索引进行排序。这在连接多个
DataFrame时很有用，可以确保索引是有序的。
copy : 布尔值，默认为None。如果为True，则不管是否需要，都会复制数据。如果为False，则尽
量避免复制数据，除非必要。None表示自动选择。'''


import pandas as pd
# 创建三个Series
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'f'])
s3 = pd.Series([7, 8, 9], index=['e', 'f', 'g'])
# 使用concat连接Series
result = pd.concat([s1, s2, s3])
print(result)