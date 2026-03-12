
#
# # count
# import pandas as pd
#
# # 创建一个包含空值的Series
# s = pd.Series([1, 2, None, 4, None])
#
# # 使用count()函数计算非空值的数量
# count_non_na = s.count()
# print(count_non_na)


# # sum
# import pandas as pd
#
# s = pd.Series([1, 2, None, 4, 5])
# total = s.sum(min_count=5)
# print(total)


#
# # mean
# import pandas as pd
#
# s = pd.Series([1, 2, None, 4, 5])
# average = s.mean(skipna=False)
# print(average)


# # median
# import pandas as pd
#
# s = pd.Series([1, 2, 3, 4, 5, 6])
# median_value = s.median()
# print(median_value)

# # min  max
# import pandas as pd
#
# s = pd.Series([1, 2, 3, 4, 5])
# min_value = s.min()
# max_value = s.max()
# print('最小值是：', min_value)
# print('最大值是：', max_value)


# # var
# import pandas as pd
# import numpy as np
#
# s = pd.Series([1, 2, np.nan, 4, 5])
# variance = s.var()
# print('方差是：', variance)
# std = s.std()
# print(std)

#
# # quantile
# import pandas as pd
#
# # 创建一个示例Series
# s = pd.Series([1, 2, 3, 4, 5])
#
# # 计算0.5分位数
# median_value = s.quantile(q=0.3)
#
# print(median_value)

#
# # cummax
# import pandas as pd
#
# # 创建一个Series
# s = pd.Series([2, 1, 3, 5, 4, 6], index=[6, 5, 4, 3, 2, 1])
#
# # 计算累积最大值
# cummax_series = s.cummax()
#
# print(cummax_series)


# # cummin
# import pandas as pd
#
# # 创建一个Series
# s = pd.Series([2, 1, 3, 5, 4, 6])
#
# # 计算累积最小值
# cummin_series = s.cummin()
#
# print(cummin_series)


# # cumsum
# import pandas as pd
#
# # 创建一个Series
# s = pd.Series([1, 2, None, 4, 5])
#
# # 计算累积和
# cumsum_series = s.cumsum()
#
# print(cumsum_series)



# cumprod
import pandas as pd

# 创建一个Series
s = pd.Series([1, 2, None, 4, 5])

# 计算累积乘积
cumprod_series = s.cumprod()

print(cumprod_series)