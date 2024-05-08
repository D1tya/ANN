import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases randomly
        self.input_layer_size = 2
        self.hidden_layer_size = 2
        self.output_layer_size = 1
        
        self.weights_input_hidden = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.bias_hidden = np.random.randn(self.hidden_layer_size)
        
        self.weights_hidden_output = np.random.randn(self.hidden_layer_size, self.output_layer_size)
        self.bias_output = np.random.randn(self.output_layer_size)
        
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
    
    def backward(self, inputs, targets, output):
        output_error = targets - output
        output_delta = output_error * self.sigmoid_derivative(output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta)
        self.bias_output += np.sum(output_delta)
        
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta)
        self.bias_hidden += np.sum(hidden_delta)
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            output = self.forward(inputs)
            self.backward(inputs, targets, output)

# Define training data (XOR function)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
nn = NeuralNetwork()
nn.train(X, y, epochs=10000)

# Test the trained model
print("Testing the trained model:")
for i in range(len(X)):
    predicted_output = nn.forward(X[i])
    print(f"Input: {X[i]}, Predicted Output: {predicted_output}")

