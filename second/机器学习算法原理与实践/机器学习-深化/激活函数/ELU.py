''''''
'''
单侧饱和函数更适合做激活函数,即类似于ReLU的激活函数
单侧饱和函数更适合做激活函数
输出值分布在0的两侧更适合做激活函数
         x ,if x > 0
f(x) = {
         α(exp(x)-1),if x<=0

f`(x) = 1 当x>0
f`(x) = f(x)+α 当x<=0      
         
'''
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def elu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=0.25):
    return np.where(x > 0, 1, alpha * np.exp(x))

# 定义x的范围
x = np.linspace(-3, 3, 100)

plt.figure(figsize=(8, 6))
plt.plot(x, relu(x), label="relu")
plt.plot(x, elu(x), label="elu (alpha=0.25)")
plt.title("relu and elu (alpha==0.25)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 6))
plt.plot(x, elu(x), label="elu (alpha==0.25)")
plt.plot(x, elu_derivative(x), label="elu_derivative (alpha==0.25)")
plt.title("elu and elu_derivative (alpha==0.25)")
plt.legend()
plt.grid(True)

plt.show()








