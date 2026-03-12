''''''
'''

f(x) = max(αx,x) 其中a是一个小常数
导数:
f`(x) = 1,当x>0时
f`(x) = α,当x<=0时

Leaky ReLU通过在负数区域引入小的正斜率α来避免ReLU的死亡问题,允许负数区域的梯度不为0
'''

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x,alpha=0.01):
    return np.where(x>0,x,alpha * x)

def leaky_relu_derivative(x,alpha=0.01):
    return np.where(x>0,1,alpha)

#定义x的范围
x = np.linspace(-3,3,100)
plt.figure(figsize=(8,6))
plt.plot(x,relu(x),label="relu")
plt.plot(x,leaky_relu(x),label="leaky_relu(alpha=0.01)")
plt.plot(x,leaky_relu_derivative(x),label='leaky_relu_derivative(alpha=0.01)')
plt.legend()
plt.grid()
plt.show()