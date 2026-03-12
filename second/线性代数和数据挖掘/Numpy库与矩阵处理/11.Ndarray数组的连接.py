'''
numpy.concatenate((a1, a2, ..., arr_n), axis=0, out=None)
(a1, a2, ..., arr_n)：这是一个包含数组的元组，这些数组需要被连接。所
有数组在除了连接轴之外的其他维度上必须有相同的形状。
axis：整数，指定沿着哪个轴进行连接。如果不指定，默认为 0，表示第一个
轴。
out：可选参数，如果提供，结果将直接存储在这个数组中。这个数组必须具有
与输入数组相同的形状和数据类型。
'''

#concatenate
# import numpy as np
# # 创建两个数组
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6]])
# print(f'数组a的形状为{a.shape}, 数组a为：\n', a)
# print(f'数组b的形状为{b.shape}, 数组b为：\n', b)
# # 沿着第一个轴（垂直方向）连接数组
# c = np.concatenate((a, b), axis=0)
# print(f'数组c的形状为{c.shape}, 数组c为：\n', c)
# # 沿着第二个轴（水平方向）连接数组
# d = np.concatenate((a, b.T), axis=1)
# print(f'数组b的转置的形状为{b.T.shape}, 数组b的转置为：\n', b.T)
# print(f'数组d的形状为{d.shape}, 数组d为：\n', d)


#stack
# import numpy as np
# # 创建三个一维数组
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(f'数组a的形状为{a.shape}, 数组a为：\n', a)
# print(f'数组b的形状为{b.shape}, 数组b为：\n', b)
# # 沿着新的轴堆叠数组
# d = np.stack((a, b), axis=0)
# print(f'数组d的形状为{d.shape}, 数组d为：\n', d)
# # 沿着第二个轴堆叠数组
# e = np.stack((a, b), axis=1)
# print(f'数组e的形状为{e.shape}, 数组e为：\n', e)


