import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        return 1 if summation > 0 else 0

    def train(self, inputs, labels, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                self.weights += self.learning_rate * error * inputs[i]

def char_to_ascii(string):
    return [ord(char) for char in string]

inputs = [
    char_to_ascii("0"),
    char_to_ascii("1"),
    char_to_ascii("2"),
    char_to_ascii("3"),
    char_to_ascii("4"),
    char_to_ascii("5"),
    char_to_ascii("6"),
    char_to_ascii("7"),
    char_to_ascii("8"),
    char_to_ascii("9")
]
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] 

perceptron = Perceptron(input_size=len(inputs[0]))
perceptron.train(inputs, labels, epochs=100)

test_inputs = [
    char_to_ascii("4"),
    char_to_ascii("5"),
    char_to_ascii("6")
]

print("Prediction for '4' (even):", perceptron.predict(test_inputs[0]))
print("Prediction for '5' (odd):", perceptron.predict(test_inputs[1]))
print("Prediction for '6' (even):", perceptron.predict(test_inputs[2]))
