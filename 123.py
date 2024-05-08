import numpy as np

class MultiLayerPerceptron:
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
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
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
    
    def predict(self, inputs):
        return self.forward(inputs)

training_data = {
    '0': np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1]]),
    '1': np.array([[0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0]]),
    '2': np.array([[1, 0, 1, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 1, 0, 1]]),
    '39': np.array([[1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 1, 1, 1]])
}

X_train = np.array([np.ravel(data) for data in training_data.values()])
y_train = np.array([[int(i == label) for i in range(4)] for label in training_data.keys()])

mlp = MultiLayerPerceptron(input_size=15, hidden_size=8, output_size=4)
mlp.train(X_train, y_train, epochs=10000)

test_data = {
    'Test 1': np.array([[1, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1]]),
    'Test 2': np.array([[1, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]])
}

print("Testing the trained model:")
for test_name, test_input in test_data.items():
    predicted_output = mlp.predict(np.ravel(test_input))
    predicted_number = np.argmax(predicted_output)
    print(f"{test_name}: Predicted number - {predicted_number}")

