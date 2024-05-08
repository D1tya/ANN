import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0

    def train(self, inputs, labels, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error

np.random.seed(0)
class_A = np.random.randn(50, 2) + [2, 2]  
class_B = np.random.randn(50, 2) + [-2, -2]  

inputs = np.concatenate((class_A, class_B))
labels = np.array([0] * 50 + [1] * 50) 

perceptron = Perceptron(input_size=2)
perceptron.train(inputs, labels, epochs=100)

plt.figure(figsize=(8, 6))

plt.scatter(class_A[:,0], class_A[:,1], color='blue', label='Class A')
plt.scatter(class_B[:,0], class_B[:,1], color='red', label='Class B')

x_values = np.linspace(-5, 5, 100)
y_values = -(perceptron.weights[0] * x_values + perceptron.bias) / perceptron.weights[1]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.title('Perceptron Decision Regions')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
