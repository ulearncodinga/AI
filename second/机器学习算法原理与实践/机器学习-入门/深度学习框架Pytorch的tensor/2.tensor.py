''''''

'''定义:
    是一种多维数组,类似于Numpy的ndarray
    是PyTorch中最基本的数据结构,用于存储和操作数据
    可以是标量,向量,矩阵,或者更高维度的数组,可以包含整数,浮点数或其他数据类型的元素'''






import torch

#打印一个标量
scalar_tensor = torch.tensor(3.14)
print("scalar_tensor:",scalar_tensor)

#打印一个向量
vector_tensor = torch.tensor([1,2,3,4,5,6])
print("vector_tensor:",vector_tensor)

#打印一个列表
matrix_tensor = torch.tensor([[1,2],[3,4]])
print("matrix_tensor:",matrix_tensor)