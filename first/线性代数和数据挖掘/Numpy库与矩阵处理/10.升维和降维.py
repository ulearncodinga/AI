#squeeze压缩维度
# import numpy as np
# # 创建一个具有单维度的数组
# arr = np.array([[[1], [2], [3]]])
# print(arr)
# print("原始数组形状:", arr.shape)
# # 使用squeeze函数去掉所有单维度的条目
# squeezed_arr = np.squeeze(arr)
# print(squeezed_arr)
# print("压缩后的数组形状:", squeezed_arr.shape)


#expend_dims增加维度
import numpy as np
# 创建一个一维数组
arr = np.array([1, 2, 3])
print("原始数组形状:", arr.shape)
print(arr)
# 在位置0增加一个维度
expanded_arr = np.expand_dims(arr, axis=0)
print("增加维度后的数组形状:", expanded_arr.shape)
print(expanded_arr)
# 在位置1增加一个维度
expanded_arr = np.expand_dims(arr, axis=1)
print("增加维度后的数组形状:", expanded_arr.shape)
print(expanded_arr)