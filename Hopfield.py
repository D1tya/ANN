import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)

    def recall(self, input_pattern, max_iterations=100):
        input_pattern = np.array(input_pattern).reshape(-1, 1)
        for _ in range(max_iterations):
            output_pattern = np.sign(np.dot(self.weights, input_pattern))
            if np.array_equal(input_pattern, output_pattern):
                return output_pattern.flatten()
            input_pattern = output_pattern
        print("Max iterations reached, recall failed.")
        return None

patterns = [
    [1, -1, 1, -1],
    [-1, -1, 1, 1],
    [1, 1, -1, -1],
    [-1, 1, -1, 1]
]

num_neurons = len(patterns[0])
hopfield_net = HopfieldNetwork(num_neurons)

hopfield_net.train(patterns)

for i, pattern in enumerate(patterns):
    recalled_pattern = hopfield_net.recall(pattern)
    if np.array_equal(pattern, recalled_pattern):
        print(f"Pattern {i+1}: Recall successful - {recalled_pattern}")
    else:
        print(f"Pattern {i+1}: Recall failed")
