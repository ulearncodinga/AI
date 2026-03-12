#mean 求平均值

# mean_arr = np.mean(arr)


# import numpy as np
#
# arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr_2d)
# mean_arr_2d = np.mean(arr_2d)
# # 输出结果
# print(mean_arr_2d)
# # 计算二维数组沿着列的平均值
# mean_arr_2d_col = np.mean(arr_2d, axis=0)
# # 输出结果
# print(mean_arr_2d_col)
# # 计算二维数组沿着行的平均值
# mean_arr_2d_row = np.mean(arr_2d, axis=1)
# # 输出结果
# print(mean_arr_2d_row)
# # 使用 keepdims=True 保留维度
# mean_arr_2d_row_keepdims = np.mean(arr_2d, axis=1, keepdims=True)
# # 输出结果
# print(mean_arr_2d_row_keepdims)

# var:方差
# std:标准差
#where 接受一个条件作为参数,返回一个元组
# np.where
import numpy as np
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
condition = arr>5
print(arr)


result = np.where(condition)
print(result)