''''''
'''
Tensor的连续性是指其元素再内存中按照其在张量中的顺序紧密存储,没有间隔
'''
import torch
a = torch.arange(12).reshape(3,4)
# print(a.flatten())
# print(a.storage().tolist())
b = a.transpose(0,1)
# print(b.flatten())
# print(b.storage().tolist())
print(b.is_contiguous())
b = b.contiguous()
print(b.is_contiguous())
