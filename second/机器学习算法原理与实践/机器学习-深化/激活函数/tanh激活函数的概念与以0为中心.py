''''''
'''
关于0中心对称,输入绝对值越大,输出越趋近+-1导数接近0
W新 = W旧 - lr x el/ef ef/ez
'''
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

#定义x的取值范围
x = np.linspace(-10,10,100)
#计算tanh
y_tanh = tanh(x)
#计算tanh的导数
y_derivative = tanh_derivative(x)


#绘制
plt.figure(figsize=(8,6))
plt.plot(x,y_tanh,label="tanh")
plt.plot(x,y_derivative,label="tanh_derivative")
plt.legend()
plt.grid()
plt.show()