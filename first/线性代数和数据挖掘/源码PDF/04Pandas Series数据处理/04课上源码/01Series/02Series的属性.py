

# # index
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=[1, 2, 3])
#
# # 打印输出该Series对象的索引。
# # Series对象的index属性可以获取到它的索引信息，这里会输出索引['a', 'b', 'c']。
# # print(series.index)
#
# # 重新给Series对象的索引赋值。
# # 将原来的索引['a', 'b', 'c']修改为新的索引['e', 'f', 'g']，
# # 这会改变Series对象中每个数据元素对应的索引标识。
# # series.index = ['e', 'f']
#
# # 再次打印输出修改索引后的Series对象，
# # print(series)
#
#
# print(series.values)



# # name
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
#
# # 打印输出该Series对象的name属性值。
# print(series.name)
#
# # 给Series对象的name属性赋值为'test'。
# # 这相当于给这个Series对象起了一个名字，方便在后续处理或展示数据时进行识别和区分。
# series.name = 'test'
#
# # 再次打印输出该Series对象的name属性值，
# print(series.name)
# print(series)


# # dtype
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'], dtype=float)
#
# # 打印输出该Series对象的数据类型。
# # Series对象的dtype属性可以获取到其数据的类型，这里创建的Series数据是整数类型，所以会输出'int64'。
# print(series.dtype)
# print(series.dtypes)
#
# # Series对象的dtype属性是只读属性，不可以直接通过赋值的方式来改变数据类型。
# # 要改变Series对象的数据类型，需要使用astype等合适的方法来进行转换操作。
# # series.dtype = 'float32'
# #
# # print(series.dtype)


# # shape
# import pandas as pd
# import numpy as np
#
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# arr = np.array([1, 2, 3])
#
#
# # 打印数组的形状
# print(series.shape)
# print(arr.shape)


# # size
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
#
# # 打印数组元素的元素数量
# print(series.size)



# empty
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3])
#
# # 判断数组是否为空
# print(series.empty)

# hasnans
# import pandas as pd
# import numpy as np
#
# # 创建Series数组
# series = pd.Series([np.nan])
# print(series)
#
# # 判断数组是否存在NaN值
# print(series.hasnans)
# print(series.empty)



# # is_unique
# import pandas as pd
# import numpy as np
#
# # 创建Series数组
# series = pd.Series(['a', 'b', 'c', np.nan, np.nan])
#
# # 判断数组中是否存在重复的元素
# print(series.is_unique)


# # nbytes
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4, 5], dtype='int64')
#
# # 获取数组元素所占用的内存大小
# print(series.nbytes)


# axes
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'], dtype='int64')
#
# # 获取数组的行轴标签
# print(series.axes)



# ndim
# import pandas as pd
#
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'], dtype='int64')
#
# # 获取数组的索引
# print(series.ndim)


# # array
# import pandas as pd
# import numpy as np
#
# # 创建一个NumPy数组
# data = np.array([1, 2, 3, 4, 5])
#
# # 使用NumPy数组创建一个Pandas Series
# series = pd.Series(data)
#
# print(series)
#
# # 打印Series的底层数据作为一个Pandas的Array对象
# print(series.array)
#
# # 打印上述Array对象的类型
# print(type(series.array))




# # attrs
# import pandas as pd
#
# # 创建一个包含整数1到5的Pandas Series
# series = pd.Series([1, 2, 3, 4, 5])
#
# # 打印额外的属性
# print(series.attrs)
#
# # 给Series添加额外的属性，这里添加了来源和时间信息
# series.attrs = {'source': 'file1', 'time': '19:27:27'}
#
# # 打印Series本身，将显示其数据和索引
# print(series)
#
# # 打印Series的额外属性
# print('额外属性', series.attrs)



# is_monotonic_decreasing
import pandas as pd

# 创建一个Pandas Series，其值从5递减到1
series = pd.Series(['a', 'b', 'c'])

# 打印检查Series是否单调递减的结果
# 由于Series中的值是递减的，所以这个表达式将返回True
print(series.is_monotonic_decreasing)
print(series.is_monotonic_increasing)