'''
在Pandas中，一维数组的创建离不开Pandas库中的Series类
'''

# class pandas.Series(data=None,index=None,dtype=None,name=None,copy=None,fastpath=False)
'''
data：可以是以下几种类型的数据：
标量值，如整数或字符串
Python列表或元组
Python字典
1d-Ndarray
index：数组或列表，用于定义Series的索引。如果未提供，则默认为从0开始的整数索引。
dtype：指定Series的数据类型。
name：给Series一个名字，用于后续的索引和操作。
copy：布尔值，默认为False。如果为True，则复制数据；如果为False，则尽可能避免复制数据，
仅影响Ndarray输入。
fastpath：布尔值，默认为False，通常不需要用户指定。它是Pandas库内部使用的一个优化标
志，当设置为 True 时，允许 Series 构造函数绕过一些检查和验证步骤，加快Series的创建速度。
但由于这个参数跳过了某些安全检查，因此在正常使用中，如果在创建Series时设置了
fastpath=True ，而传入的数据又不符合预期，则可能会导致不可预测的行为或错误。
'''


#使用标量创建相同元素的Series
# import pandas as pd
# import numpy as np
#
# data = 5,6,5,4
#
# # 使用pandas库的Series函数创建一个Series对象
# # 第一个参数是要作为Series数据的值，这里是0
# # 第二个参数是一个列表，用于指定Series的索引，这里指定了索引为['a', 'b', 'c']
# series = pd.Series(data,index=['a','b','c','z'],name='Test',dtype=np.int32)
#
# print(series)
# print(type(series))



'''
使用列表/元组创建
'''
import pandas as pd
#
# data1 = [1,2,3,4,5]
# data2 = (1,2,3,4,5)
#
# series1 = pd.Series(data1,index = ['a','b',1,2,3])
# series2 = pd.Series(data2,index = ['a','b',1,2,3])
# print(series1)
# print(series2)


'''
使用字典创建Series
使用字典创建Series时，字典的键就是索引，字典的值就是该索引对应的值。如果使用字典创建
Series，并且指定了与字典的键不同的index参数，那么生成的Series数组中的数据就是以index参数的
值为索引，但索引所对应的值是NaN。
在Pandas中， NaN （Not a Number）是一个特殊的浮点数，用于表示缺失数据或无效数据。 NaN 是
IEEE 浮点标准的一部分，Pandas 使用 NaN 来表示数据集中缺失或未定义的值。
'''
# import pandas as pd
# # 定义一个字典data，其中包含了三个键值对，键分别为'a'、'b'、'c'，对应的值分别为1、2、3
# data = {'a': 1, 'b': 2, 'c': 3}
# # 使用pandas库的Series函数创建一个Series对象。
# # 当传入一个字典作为参数时，字典的键会自动成为Series的索引，字典的值会成为对应索引下的数值。
# series = pd.Series(data)
# # 打印输出创建好的Series对象
# print(series)








'''
使用Ndarray创建Series
'''

import pandas as pd
import numpy as np
# 使用numpy的array函数创建一个一维数组data，数组中包含了整数1到5。
data = np.array([1, 2, 3, 4, 5])
# 使用pandas库的Series函数创建一个Series对象。


# 传入刚刚创建的一维numpy数组data作为参数，此时会将该数组的数据依次作为Series的数值，
# 并且会自动生成一个默认的整数索引（从0开始，依次递增，与数组元素的下标对应）。
series = pd.Series(data,index = ['a','b','c','d','e'],copy=True)
#这里默认copy=False,为False时原数组会受影响,为True时不受影响(在列表时不管说什么都不受影响)


series[0] = 100

# 打印输出创建好的Series对象，以便查看其具体内容，包括索引和对应的数据值等信息。
print(series)

