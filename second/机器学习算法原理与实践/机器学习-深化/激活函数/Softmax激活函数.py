''''''
'''
概率分布:Softmax 函数将输入转换为概率分布,
因此在多分 类问题中常用于将模型的原始输出转换为概率值｡ 

连续可导:Softmax 函数是连续可导的,
这使得它可以与梯度 下降等优化算法一起使用进行训练｡

概率分布:Softmax 函数将输入转换为概率分布,
因此在多分 类问题中常用于将模型的原始输出转换为概率值｡ 

连续可导:Softmax 函数是连续可导的,
这使得它可以与梯度下降等优化算法一起使用进行训练｡

'''

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    vals = np.exp(x)
    return vals / np.sum(vals)

def softmax_derivative(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s,s)

x = np.linspace(-5,5,100)
y_softmax = softmax(x)
y_derivative = softmax_derivative(x)

plt.figure(figsize=(8,6))
plt.plot(x,y_softmax)
plt.title("softmax")

plt.figure(figsize=(8,6))
plt.plot(x,y_derivative)
plt.title("softmax_derivative")

plt.grid()
plt.legend()
plt.show()