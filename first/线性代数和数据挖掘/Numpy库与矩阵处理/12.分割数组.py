#split
# import numpy as np
# # 创建一个一维数组
# a = np.array([1, 2, 3, 4, 5, 6])
# # 使用 split 平均分割数组为 3 个子数组
# result1 = np.split(a, 3)
# # 使用 split 按位置分割数组
# result2 = np.split(a, [2, 4])
# # 输出结果
# print("平均分割为 3 个子数组：", result1)
# print("按位置分割：", result2)
import numpy

#hsplit
# numpy.hsplit
# import numpy as np
# arr = np.arange(24).reshape(6,4)
#
# #打印原始数组
# print("原始数组:")
# print(arr)
#
# #指定每部分应该包含的列数
# col_counts = [1,2]
#
# #使用hsplit分割数组
# subarrays = np.hsplit(arr,col_counts)
#
# #打印分割后的子数组
# print("\n分割后的子数组")
# print(subarrays)








#vsplit
# numpy.vsplit(ary,indices_or_sections)

# ary:要分割的数组,通常是个二维数组
# indices_or_sections:可以是一个整数,表示要将数组平均分割多少个子数组;也可以是一个整数数组,表示分割的位置
import numpy as np
# 创建一个 6x4 的二维数组
arr = np.arange(24).reshape(6, 4)
# 打印原始数组
print("原始数组:")
print(arr)
# 指定每部分应该包含的列数
col_counts = [1, 2]
# 使用 hsplit 分割数组
subarrays = np.vsplit(arr, col_counts)
# 打印分割后的子数组
print("\n分割后的子数组:")
print(subarrays)
