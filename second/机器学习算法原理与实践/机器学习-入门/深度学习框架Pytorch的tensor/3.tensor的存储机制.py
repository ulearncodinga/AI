''''''
import torch

'''
tensor包含两部分:storage和metadata
Storage(存储)
    一维数组,存储tensor元素值
    
    
Metadata(元数据)
    tensor的形状,数据类型,步幅,存储偏移量,设备
    
'''

# tensor1= torch.tensor([[1,2,3],[4,5,6]])
# #打印张量
# print(tensor1)
# #打印形状
# print(tensor1.shape)
# #打印数据类型
# print(tensor1.dtype)
# #存储的内容
# print(tensor1.storage().tolist())


'''
storage的存储
'''
a = torch.arange(12).reshape(3,4)
print(a)
# print(a.storage().tolist())
print(a.storage)
b = a.transpose(0,1)
print(b)
# print(b.storage().tolist())


'''
tensor的步长
'''
a = torch.arange(12).reshape(3,4)
print(a)
print(a.stride())

'''tensor的偏移
是指张量的第一个元素开始的索引位置'''
a = torch.arange(12).reshape(3,4)
print(a)
b = a[1:3,0:2]
print(b)