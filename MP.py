class MPNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        weighted_sum = sum([inputs[i] * self.weights[i] for i in range(len(inputs))])
        return 1 if weighted_sum <= self.threshold else 0

def ANDNOT(input1, input2):
    weights = [1, -1]  
    threshold = 0

    neuron = MPNeuron(weights, threshold)
    
    return neuron.activate([input1, input2])

print("ANDNOT(0, 0) =", ANDNOT(0, 0))
print("ANDNOT(0, 1) =", ANDNOT(0, 1))
print("ANDNOT(1, 0) =", ANDNOT(1, 0))
print("ANDNOT(1, 1) =", ANDNOT(1, 1))
