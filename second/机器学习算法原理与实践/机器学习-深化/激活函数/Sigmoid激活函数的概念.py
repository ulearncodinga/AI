''''''
'''
Sigmoid函数的输出范围被限制在0-1之间的
Sigmoid的导数两端趋近于0

'''
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


#定义x的范围
x = np.linspace(-10,10,100)
#计算sigmoid的值
y_sigmoid = sigmoid(x)
# 计算sigmoid导数的值
y_derivative = sigmoid_derivative(x)

#绘制
plt.figure(figsize=(8,6))
plt.plot(x,y_sigmoid,label="sigmoid")
plt.plot(x,y_derivative,label="sigmoid_derivative")
plt.title("sigmoid and sigmoid_derivative")
plt.legend()
plt.grid()
plt.show()