# 属性
'''
1. index 返回Series中的索引
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# # 打印输出该Series对象的索引。
# # Series对象的index属性可以获取到它的索引信息，这里会输出索引['a', 'b', 'c']。
# print(series.index)
# # 重新给Series对象的索引赋值。
# # 将原来的索引['a', 'b', 'c']修改为新的索引['e', 'f', 'g']，
# # 这会改变Series对象中每个数据元素对应的索引标识。
# series.index = ['e', 'f', 'g']
# # 再次打印输出修改索引后的Series对象，
# print(series)

'''
2. values 返回Series中的数据,返回的数据以Ndarray的形式存在
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# # 打印输出该Series对象的数据。
# print(series.values)
# print(type(series.values))
# # # values无法通过直接赋值的方式去修改
# # series.values = ['e', 'f', 'g']
# # print(series)




'''
3 name 返回Series的名称
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# # 打印输出该Series对象的name属性值。
# print(series.name)
# # 给Series对象的name属性赋值为'test'。
# # 这相当于给这个Series对象起了一个名字，方便在后续处理或展示数据时进行识别和区分。
# series.name = 'test'
# # 再次打印输出该Series对象的name属性值，
# print(series.name)





'''
4. dtype 和 dtypes 查看返回Series对象的数据类型,在Series中两个作用一样(只读属性)
'''
import pandas as pd
# 创建Series数组
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# 打印输出该Series对象的数据类型。
# Series对象的dtype属性可以获取到其数据的类型，这里创建的Series数据是整数类型，所以会输出'int64'。
print(series.dtype)
# Series对象的dtype属性是只读属性，不可以直接通过赋值的方式来改变数据类型。
# 要改变Series对象的数据类型，需要使用astype等合适的方法来进行转换操作。
# series.dtype = 'float32'
#
# print(series.dtype)


'''
5. shape 描述形状
'''
# import pandas as pd
# import numpy as np
#
# series = pd.Series([1,2,3,4],['a','b','c','d'])
# arr = np.array([1,2,3])
#
# print(series.shape)
# print(arr.shape)
'''
6.size
用于返回Series的元素数量，该返回值是一个整数。
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# # 打印数组元素的元素数量
# print(series.size)

'''
7.empty
用来表示Series数组是否为空，返回值一个布尔值，如果数组里一个元素都没有就返回True，否则返回
False。
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series()
# # 判断数组是否为空
# print(series.empty)

'''
8 hasnans
用于返回数组中是否包含NaN值，如果数组中存在NaN，那么返回True，否则返回False。
'''
# import pandas as pd
# import numpy as np
# 创建Series数组
# series = pd.Series([1, 2, 3, np.nan], index=['a', 'b', 'c', 'd'])
# 判断数组是否存在NaN值
# print(series.hasnans)


'''
9 is_unique
用于返回数组中的元素是否为独一无二的，如果所有的元素都是独一无二的，即数组中没有重复元素，
那么就返回True，否则返回False。
'''
# import pandas as pd
#
# series = pd.Series(['a','b','c'])
# print(series.is_unique)

'''
10 nbytes
用于返回该Series对象中所有数据占用的总字节数。
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4, 5], dtype='int64')
# # 获取数组元素所占用的内存大小
 # print(series.nbytes)

'''
11. axes
用于返回series对象行轴标签的列表
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'],
# dtype='int64')
# # 获取数组的行轴标签
# print(series.axes)

'''
12.ndim
返回Series数组的维度，对于Series数组来说，它的维度始终为1
'''
# import pandas as pd
# # 创建Series数组
# series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'],
# dtype='int64')
# # 获取数组的索引
# print(series.ndim)
'''.13 array
用于返回Series的底层数组，包括数组的元素、数组的长度及数组元素的数据类型。
'''
import pandas as pd
import numpy as np
# 创建一个NumPy数组
data = np.array([1, 2, 3, 4, 5])
# 使用NumPy数组创建一个Pandas Series
series = pd.Series(data)
# 打印Series的底层数据作为一个Pandas的Array对象
print(series.array)
# 打印上述Array对象的类型
print(type(series.array))

'''
.14 attrs
返回series的自定义属性，可以用来存储额外的说明性数据
'''
# import pandas as pd
# # 创建一个包含整数1到5的Pandas Series
# series = pd.Series([1, 2, 3, 4, 5])
# # 打印额外的属性
# print(series.attrs)
# # 给Series添加额外的属性，这里添加了来源和时间信息
# series.attrs = {'source': 'file1', 'time': '19:27:27'}
# # 打印Series本身，将显示其数据和索引
# print(series)
# # 打印Series的额外属性
# print('额外属性', series.attrs)

'''
15 is_monotonic_decreasing
返回一个布尔值，表示Series是否按降序排列。

'''
# import pandas as pd
# # 创建一个Pandas Series，其值从5递减到1
# series = pd.Series([5, 4, 3, 2, 1])
# # 打印检查Series是否单调递减的结果
# # 由于Series中的值是递减的，所以这个表达式将返回True
# print(series.is_monotonic_decreasing)


'''
16 is_monotonic_increasing
返回一个布尔值，表示Series 是否按升序排列。
'''
import pandas as pd
# 创建一个Pandas Series，其值从1递增到5
series = pd.Series([1, 2, 3, 4, 5])
# 打印检查Series是否单调递增的结果
# 由于Series中的值是递增的，所以这个表达式将返回True
print(series.is_monotonic_increasing)













