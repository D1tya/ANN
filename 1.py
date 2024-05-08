import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

x = np.linspace(-5, 5, 100)


y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_softmax = softmax(np.array([x, x/2, -x]).T)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')

plt.subplot(2, 2, 2)
plt.plot(x, y_relu)
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')

plt.subplot(2, 2, 3)
plt.plot(x, y_tanh)
plt.title('Hyperbolic Tangent (Tanh) Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')

plt.subplot(2, 2, 4)
plt.plot(x, y_softmax)
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')

plt.tight_layout()
plt.show()
