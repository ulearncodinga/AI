import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def Prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)


def Prelu_derivative(x, alpha=0.25):
    return np.where(x > 0, 1, alpha)

# 定义x的范围
x = np.linspace(-3, 3, 100)

plt.figure(figsize=(8, 6))
plt.plot(x, relu(x), label="relu")
plt.plot(x, Prelu(x), label="Prelu (alpha=0.25)")
plt.title("relu and Prelu (alpha==0.25)")
plt.legend()
plt.grid(True)

plt.figure(figsize=(8, 6))
plt.plot(x, Prelu(x), label="Prelu (alpha==0.25)")
plt.plot(x, Prelu_derivative(x), label="Prelu_derivative (alpha==0.25)")
plt.title("Prelu and Prelu_derivative (alpha==0.25)")
plt.legend()
plt.grid(True)

plt.show()




