''''''
'''
Parametric ReLu

f(x) = max(αx,x),其中α是一个可学习参数
f`(x) = 1,当x>0
f`(x) = α,当x<=0
'''
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0,x)

def Prelu(x,alpha=0.01):
    return np.where(x>0,x,alpha * x)

def Prelu_derivative(x,alpha=0.01):
    return np.where(x>0,1,alpha)

#定义x的范围
x = np.linspace(-3,3,100)
plt.figure(figsize=(8,6))
plt.plot(x,relu(x),label="relu")
plt.plot(x,Prelu(x),label="Prelu(alpha=0.01)")
plt.plot(x,Prelu_derivative(x),label='Prelu_derivative(alpha=0.01)')
plt.legend()
plt.grid()
plt.show()