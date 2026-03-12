

# 使用标量创建
# import pandas as pd
# import numpy as np
#
# data = 5
#
# series = pd.Series(data, index=['a', 'b', 'c'], name='test')
#
# print(series)
# print(type(series))


# # 使用列表创建
# import pandas as pd
#
# data1 = (1, 2, 3, 4, 5)
#
# series1 = pd.Series(data1, index=['a', 'b', 1, 3, 1])
# print(series1)

#
# # 使用字典创建
# import pandas as pd
#
# # 定义一个字典data，其中包含了三个键值对，键分别为'a'、'b'、'c'，对应的值分别为1、2、3
# data = {'a': 1, 'b': 2, 'c': 3}
#
# # 使用pandas库的Series函数创建一个Series对象。
# # 当传入一个字典作为参数时，字典的键会自动成为Series的索引，字典的值会成为对应索引下的数值。
# series = pd.Series(data, index=['x'])
#
# # 打印输出创建好的Series对象
# print(series)




# ndarray
import pandas as pd
import numpy as np

# 使用numpy的array函数创建一个一维数组data，数组中包含了整数1到5。
data = [1, 2, 3, 4, 5]

# 使用pandas库的Series函数创建一个Series对象。
# 传入刚刚创建的一维numpy数组data作为参数，此时会将该数组的数据依次作为Series的数值，
# 并且会自动生成一个默认的整数索引（从0开始，依次递增，与数组元素的下标对应）。
series = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'], copy=False)

series[0] = 100

# 打印输出创建好的Series对象，以便查看其具体内容，包括索引和对应的数据值等信息。
print(series)
print(data)