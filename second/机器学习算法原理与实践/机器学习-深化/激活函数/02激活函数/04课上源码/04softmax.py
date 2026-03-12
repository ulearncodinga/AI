import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    vals = np.exp(x)
    return vals / np.sum(vals)


def softmax_derivative(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)

x = np.linspace(-5, 5, 100)
y_softmax = softmax(x)
y_derivative = softmax_derivative(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y_softmax)
plt.title("softmax")

plt.figure(figsize=(8, 6))
plt.plot(x, y_derivative)
plt.title("softmax_derivative")

plt.show()














