import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 定义x的范围
x = np.linspace(-10, 10, 100)
# 计算relu的值
y_relu = relu(x)
# 计算relu导数的值
y_derivative = relu_derivative(x)

# 绘制
plt.figure(figsize=(8, 6))
plt.plot(x, relu(x), label="relu")
plt.plot(x, relu_derivative(x), label="relu_derivative")
plt.title("relu and relu_derivative")
plt.legend()
plt.grid(True)
plt.show()








