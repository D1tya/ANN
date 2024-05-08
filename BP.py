import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output
    
    def backward(self, inputs, targets):
        output_error = targets - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta) * self.learning_rate
        
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta) * self.learning_rate
        
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            self.forward(inputs)
            self.backward(inputs, targets)

input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

nn.train(X, y, epochs=10000)

for i in range(len(X)):
    predicted_output = nn.forward(X[i])
    print(f"Input: {X[i]}, Predicted Output: {predicted_output}")
