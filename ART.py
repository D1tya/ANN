import numpy as np

class ART:
    def __init__(self, num_input, vigilance_parameter=0.9):
        self.num_input = num_input
        self.vigilance_parameter = vigilance_parameter
        self.weights = np.random.rand(num_input)
        self.bias = np.random.rand()

    def normalize_input(self, input_pattern):
        return input_pattern / np.sum(input_pattern)

    def calculate_similarity(self, input_pattern):
        normalized_input = self.normalize_input(input_pattern)
        return np.dot(self.weights, normalized_input)

    def resonance(self, input_pattern):
        similarity = self.calculate_similarity(input_pattern)
        while similarity < self.vigilance_parameter:
            self.weights += input_pattern
            self.bias += 1
            similarity = self.calculate_similarity(input_pattern)
        return similarity, self.weights

num_input = 5
vigilance_parameter = 0.8

art = ART(num_input, vigilance_parameter)

input_pattern = np.array([1, 1, 0, 1, 0])

similarity, weights = art.resonance(input_pattern)

print("Similarity:", similarity)
print("Updated weights:", weights)
