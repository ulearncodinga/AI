# Ndarray数组中的元素是可以被修改的，如果需要访问或者修改Ndarray数组某个位
# 置的元素，则需要使用Ndarray数组的索引来完成；如果需要访问或者修改一些区域
# 的元素，则需要使用Ndarray数组的切片。
import numpy as np

#创建一个一维数组
# arr = np.array([1, 2, 3, 4, 5])
# print(arr[0])
# print(arr[4])
# print(arr[-1])
'''
切片
    与Python列表的切片方式类似，使用冒号 :进行切片，格式为 start:stop:step，
其中start、stop、step可根据情况省略。
'''
# import numpy as np
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# arr1 = arr[4:7]
# print('arr1数组元素为', arr1)
# arr2 = arr[1:6:2]
# print('arr2数组元素为', arr2)
# 将一个标量值赋值给一个切片时，该值会自动传播到整个选区。
# arr[4:7] = 6
# print('arr数组元素为：', arr)
# 也可以使用:代表全部元素
# arr[:] = 10
# print('arr数组元素为：', arr)








'''
多维数组的索引 
 
'''
#
# import numpy as np
#
# arr = np.array([[1,2,3],[4,5,6]])
# print('arr的数组为:\n',arr)
# print('arr[0]为: ',arr[0])
# print('arr[0][1]为; ',arr[0][1])
# print('arr[0,1]为: ',arr[0,1])
#超出范围会报错 IndexError 索引异常
'''
多维数组的切片

'''

import numpy as np
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('arr:\n',arr)

#选择特定的一行
row1 = arr[1]
print(row1)

#选择连续的多行
rows = arr[2:3]
print(rows)

#选择不连续的多行
rows = arr[[0,2]]
print(rows)

#选择特定的一列
col1 = arr[:,0]#只要第零列
print(col1)
#选择连续的多列
cols = arr[:,1:3]
print(cols)
#选择不连续的多列
cols = arr[:,[0,2]]
print(cols)


# 同时选择行和列
subset = arr[1:3,1:3]
print(subset)