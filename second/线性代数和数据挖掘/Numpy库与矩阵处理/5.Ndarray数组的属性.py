'''
1.shape:查看Ndarray数组的形状
'''
import numpy as np
arr1 = np.array([[1,2,3],[4,5,6]])
print(arr1)
print(arr1.shape)#打印形状

'''
2.dtype 查看Ndarray数组的数据类型,调用Ndarray.dtype
'''
import numpy as np
arr1 = np.array([[1,2,3],[4,5,6]])
print(arr1)
print(arr1.dtype)

'''
size:数组中包含元素的个数,,其大小等于调用shape函数返回的元组中的所有元素的乘积
'''
'''
ndim:数组维度的大小,其大小等于调用shape函数返回的元组中的元素的个数
'''
import numpy as np
arr1 = np.array([[1,2,3],[4,5,6]])
print(arr1)
print(arr1.shape)
print(arr1.ndim)