'''
arange函数
numpy.arange([start],stop[,step,],dtype=None)
返回的是一维数组
'''
#使用arange函数创建Ndarray数组


# import numpy as np
# # 创建一个从0到9的数组
# arr1 = np.arange(10)
# print(arr1)
# # 创建一个从5到14的数组，步长为2
# arr2 = np.arange(5, 15, 2)
# print(arr2)
# # 创建一个从0到1的数组，包含10个值（步长为0.1）
# arr3 = np.arange(0, 1, 0.1)
# print(arr3)
# # 指定数据类型
# arr4 = np.arange(10, dtype=np.float32)
# print(arr4)


#zeros函数
#创建指定长度或者形状的全0数组,函数原型为:
# numpy.zeros(shape,dtype=float,order='C')

#创建一维全0数组
# import numpy as np
# arr1 = np.zeros(5)
# print(arr1)

#使用zeros创建二维全0数组
# import numpy as np
# arr2 = np.zeros((3,2))
# print(arr2)


# 创建三维全0数组
# import numpy as np
# arr3 = np.zeros((3,2,3))
# print(arr3)






#ones函数
# 填充的元素是1


#empty函数
# 创建未初始化数组,创建一个指定形状,数据类型且未初始化的数组
# numpy.empty(shape, dtype=float, order='C')


# import numpy as np
# x = np.empty((2,4))
# print(x)
# y = np.empty((2,3),dtype=int)
# print(y)
#



#full函数
#创建指定形状,指定元素的数组
# numpy.full(shape, fill_value, dtype=None, order='C')


import numpy as np
x = np.full((2,3),7)
print(x)

y = np.full((2,3),5.5,dtype=int)
print(y)