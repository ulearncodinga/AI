'''
数据类型的修改:astype


numpy.ndarray.astype(dtype, order='K', casting='unsafe',
subok=True, copy=True)

dtype：表示新数据类型的对象。这可以是 NumPy 的 dtype 对象，也可以是
Python 的数据类型，例如 int 或 float。
order：一个字符，指定结果的内存布局。 'C' 表示按行（C风格）， 'F' 表示
按列（Fortran风格）， 'A' 或 'K' 表示在内存中的顺序与原数组保持一致。
casting：控制数据类型转换的安全性。它可以是
'no'、 'equiv'、 'safe'、 'same_kind' 或 'unsafe' 之一。
subok：如果为 True，则子类将被传递，否则返回的数组将被强制转换为基类
数组。
copy：布尔值，指定是否复制数据。
'''
# import numpy as np
# # 创建一个浮点数数组
# arr = np.array([1.1, 2.2, 3.3])
# # 使用 astype 方法将数组转换为整数类型
# new_arr = arr.astype(np.int32)
# print("Original array:", arr)
# print("old type:", arr.dtype)
# print("New array:", new_arr)
# print("new type:", new_arr.dtype)


'''
形状的修改
reshape
numpy.ndarray.reshape(newshape, order='C')

newshape：整数或整数元组，新的形状应该与原始数组中的元素数量相匹配。
order：可选参数，‘C’ 表示按行（C-style），‘F’ 表示按列（Fortran-style），‘A’ 
表示原数组内存顺序，‘K’ 表示元素在内存中的出现顺序。
'''

# import numpy as np
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = a.reshape((3, 2))
# print(a)
# print('shape--a', a.shape)
# print(b)
# print('shape--b',b.shape)

'''
resize

resize 方法用于改变数组的大小，与 reshape类似，但它会直接修改调用它的原始数
组。如果新形状大于原始形状，则会在数组的末尾添加新的元素，这些元素的值未定
义；如果新形状小于原始形状，则会截断数组。函数原型:
    numpy.ndarray.resize(newshape)
'''
# import numpy as np
# a = np.array([[1, 2, 3], [4, 5, 6]])
# print('before:', a.shape)
# a.resize((3, 2))
# print('after:', a.shape)
# a.resize((2, 2))
# print(a)
# a.resize((5, 5))
# print(a)

'''
flatten
flatten方法返回一个一维数组，它是原始数组的拷贝，它默认按行顺序展平数组，但
可以通过参数 order 来指定展平顺序。函数原型为

numpy.ndarray.flatten(order='C')
'''
# import numpy as np
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = a.flatten()
# print(b)
'''
ravel
ravel方法返回一个连续的数组，它尝试以最低的复制操作来返回展平后的数组，函
数原型为：
numpy.ndarray.ravel(order='C')
'''
# import numpy as np
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = a.ravel()
# print(b)

'''

T
Ndarray 对象的 T 属性是一个特殊的属性，它返回数组的转置（transpose）。
转置是一个操作，它将数组的维度进行交换。对于二维数组，这意味着行变为列，列
变为行。对于更高维度的数组，转置操作涉及到交换数组的轴
'''
import numpy as np
# 创建一个二维数组
arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
# 获取转置
transposed_arr = arr_2d.T
print("Original array:")
print(arr_2d)
print("Transposed array:")
print(transposed_arr)

