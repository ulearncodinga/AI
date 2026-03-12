#append
# 用于将值追加到数组末尾
# numpy.append(arr,values,axis=None)
'''arr :原始数组
values：要追加的值，它们会被追加到 arr 的末尾。 values 的形状必须与
arr 在除了要追加的轴之外的所有轴上兼容。
axis：可选参数，指定要追加值的轴。如果 axis 没有被指定， arr 会被展
平，values 会被追加到结果数组的末尾。'''
import numpy as np
# 创建一个一维数组
arr = np.array([1, 2, 3])
# 要追加的值
values = np.array([4, 5, 6])
# 使用 numpy.append 追加值
result = np.append(arr, values)
# 输出结果
print(result)
# 创建一个二维数组
arr_2d = np.array([[1, 2], [3, 4]])
# 要追加的值
values_2d = np.array([[5], [6]])
# 使用 numpy.append 追加值，沿着列方向
result_2d = np.append(arr_2d, values_2d, axis=1)
# 输出结果
print(result_2d)

'''
insert
该函数用于在数组的指定位置插入元素。与 numpy.append 不同的是，
numpy.insert 允许用户在数组的任意位置插入元素，而不仅仅是数组的末尾。函数
原型为：
numpy.insert(arr, obj, values, axis=None)
'''
'''
delete
该函数用于从数组中删除指定的子数组，并返回一个新的数组。这个函数允许用户删
除数组中的单个元素或多个连续的元素。函数原型为：

numpy.delete(arr, obj, axis=None)
'''