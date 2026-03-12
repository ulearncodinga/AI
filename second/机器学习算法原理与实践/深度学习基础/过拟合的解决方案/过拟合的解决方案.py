''''''
'''
1.正则化
2.Dropout在神经网络训练过程中使用的正则化技术
为什么Dropout可以解决过拟合问题:
    减少过拟合:随即丢弃神经元,迫使网络学习对于任何单个神经元的变化都要更加鲁棒的特征表示
    取平均的作用:通过随机丢弃神经元,每次前向传播都相当于在训练不同的子网络
                在测试阶段,不再进行Dropout,但是通过保留所有的权重,网络结构变得更加完整
'''
import torch



pre_layer_out = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)

m = torch.nn.Dropout(p=0.2)
input = torch.randn(10,1)
# print(input)
output = m(input)
# print(output)
def dropout_layer(p):
    global input,pre_layer_out
    if p==1:
        return torch.zeros_like(pre_layer_out)
    if p==0:
        return input
    mask = (input>p).float()
    print(mask)
    return mask * pre_layer_out / (1.0 - p)

print(dropout_layer(0.2))