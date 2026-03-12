import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# 定义x的范围
x = np.linspace(-3, 3, 100)

plt.figure(figsize=(8, 6))
plt.plot(x, relu(x), label="relu")
plt.plot(x, leaky_relu(x), label="leaky_relu (alpha=0.01)")
plt.title("relu and leaky_relu (alpha=0.01)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 6))
plt.plot(x, leaky_relu(x), label="leaky_relu (alpha=0.01)")
plt.plot(x, leaky_relu_derivative(x), label="leaky_relu_derivative (alpha=0.01)")
plt.title("leaky_relu and leaky_relu_derivative (alpha=0.01)")
plt.legend()
plt.grid(True)

plt.show()




