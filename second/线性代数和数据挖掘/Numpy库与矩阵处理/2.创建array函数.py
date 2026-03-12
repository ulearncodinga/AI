'''
创建Ndarray数组最简单的方式就是使用array函数，它接受一切序列型的对象，然
后产生一个新的含有传入数据的Numpy数组。
'''
# numpy.array(object,dtype=None,copy=True,order="K",subok=False,ndmin=0)
'''object:可以是列表、元组、数组等可迭代对象，表示要转换为数组的数据。
2. dtype:可以指定数组的数据类型，如果未指定，则根据输入数据自动推断。
3. copy:如果为True，则复制输入数据；如果为False，则只有必要时才复制，这意
味着如果 object 已经是一个 NumPy 数组并且满足其他条件（如 dtype、order
等），则不会进行复制。默认为True。
4. order:指定数组在内存中的存储顺序，可以是‘C’（C语言风格,行优
先）、‘F’（Fortran风格,列优先）或‘K’（尽可能与输入数据的顺序一致）。默认
为‘K’。
5. subok:如果为True，则返回的数组将是输入对象的子类；如果为False，则返回
的数组将始终是基类数组。默认值为False。
6. ndmin:指定数组的最小维度。如果输入数据的维度小于ndmin，则 NumPy 会自
动在前面添加维度，以使数组的维度至少为ndmin'''

#创建一维数组
# import numpy as np
# arr1 = np.array([1,2,3,4,5])
# print(type(arr1))
# print(arr1)


#创建二维数组
# import numpy as np
#
# arr1 = np.array([
#     [1,2,3],
#     [4,5,6]
# ])
# print(type(arr1))
# print(arr1)


#创建三维数组
import numpy as np

arr1 = np.array([
    [[1,2,3],
    [4,5,6]],

    [[5,6,7],
    [4,5,6]]

])
print(type(arr1))
print(arr1)














