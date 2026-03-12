# count 用来计算Series中非NaN(非空)值的数量
# import pandas as pd
# # 创建一个包含空值的Series
# s = pd.Series([1, 2, None, 4, None])
# # 使用count()函数计算非空值的数量
# count_non_na = s.count()
# print(count_non_na)






#sum
# 计算所有值的总和
# Series.sum(axis=None, skipna=True, numeric_only=None, min_count=0)
'''
axis : 对于 Series 对象来说，这个参数通常不起作用，因为 Series 是一维的。它主要在
DataFrame 对象中用于指定操作的轴（0表示按列求和，1表示按行求和）。
skipna : 布尔值，默认为 True 。如果为 True ，则在计算总和时会忽略 NaN 值，如果为False，则
返回NaN。
numeric_only : 布尔值，默认为 None 。如果为 True ，则只对数字类型的数据进行计算，只针对
DataFrame。
min_count : int值，默认为0。表示在计算总和之前，至少需要多少个非 NaN 值。如果非 NaN 值的
数量小于 min_count ，则结果为 NaN 。
# '''
# import pandas as pd
# s = pd.Series([1, 2, None, 4, 5])
# total = s.sum(  )
# print(total)


'''
mean
mean() 函数会计算所有值的平均值。


Series.mean(axis=None, skipna=True, numeric_only=None)

axis : 对于 Series 对象来说，这个参数通常不起作用，因为 Series 是一维的。它主要在
DataFrame 对象中用于指定操作的轴（0表示按列计算平均值，1表示按行计算平均值）。
skipna : 布尔值，默认为 True 。如果为 True ，则在计算平均值时会忽略 NaN 值，如果为False，
则返回NaN。
numeric_only : 布尔值，默认为 None 。如果为 True ，则只对数字类型的数据进行计算，只针对
DataFrame。
'''
# import pandas as pd
# s = pd.Series([1, 2, 3, 4, 5])
# average = s.mean()
# print(average)


'''
4 median
median() 函数用于计算DataFrame或Series中的中位数
Series.median(axis=0, skipna=True, numeric_only=False)
# '''
# import pandas as pd
# s = pd.Series([1, 2, 3, 4, 5])
# median_value = s.median()
# print(median_value)


'''
5 min和max
Series.min() 函数用于计算Series对象中的最小值， Series.max() 函数用于计算Series对象中的最大
值。
'''


'''
6 var
Series.var() 函数用于计算Series对象的方差。

Series.var(axis=None, skipna=True, ddof=1, numeric_only=False)

axis : 对于 Series 对象，这个参数不会产生任何效果，因为 Series 是一维的。在 DataFrame
中， axis 用于指定沿着哪个轴计算方差。
skipna : 布尔值，默认为 True 。如果为 True ，则在计算方差之前会忽略 NaN 值。如果设置为
False ，计算方差时会包括 NaN 值，通常会导致结果也是 NaN 。
ddof : 整数，默认为 1 。Delta Degrees of Freedom，用于贝塞尔校正，以得到样本方差的估
计。对于无偏估计（样本方差）， ddof 通常设置为 1 。如果计算总体方差，应该将 ddof 设置为
0 。
numeric_only : 布尔值，默认为 False 。如果为 True ，则只对数字类型的数据进行方差计算，忽
略非数字类型的数据。
'''


'''
7 std
Series.std() 函数用于计算Series对象的标准差

Series.std(axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs)
'''
# import pandas as pd
# import numpy as np
# s = pd.Series([1, 2, np.nan, 4, 5])
# std_dev = s.std()
# print('标准差是：', std_dev)



'''

cummax
该方法用于计算Series中元素的累积最大值，返回一个相同长度的Series，其中每个位置上的值表示从
Series开始到当前位置（包括当前位置）的最大值。

Series.cummax(axis=None, skipna=True, *args, **kwargs)
'''
# import pandas as pd
# # 创建一个Series
# s = pd.Series([2, 1, 3, 5, 4, 6])
# # 计算累积最大值
# cummax_series = s.cummax()
# print(cummax_series)

'''
0 cummin
该方法用于计算Series中元素的累积最小值，返回一个相同长度的Series，其中每个位置上的值表示从
Series开始到当前位置（包括当前位置)的最小值

Series.cummin(axis=None, skipna=True, *args, **kwargs)
'''
# import pandas as pd
# # 创建一个Series
# s = pd.Series([2, 1, 3, 5, 4, 6])
# # 计算累积最小值
# cummin_series = s.cummin()
# print(cummin_series)