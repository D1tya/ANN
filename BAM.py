import numpy as np

class BAM:
    def __init__(self, pattern1, pattern2):
        self.pattern1 = pattern1
        self.pattern2 = pattern2
        self.W = np.outer(pattern1, pattern2)

    def retrieve_pattern1(self, input_pattern):
        return np.dot(input_pattern, self.pattern2)

    def retrieve_pattern2(self, input_pattern):
        return np.dot(input_pattern, self.pattern1)

pattern1 = np.array([1, 0, 1])
pattern2 = np.array([0, 1, 0])
pattern3 = np.array([1, 1, 1])

bam = BAM(pattern1, pattern2)

retrieved_pattern1 = bam.retrieve_pattern1(pattern3)
retrieved_pattern2 = bam.retrieve_pattern2(pattern3)

print("Retrieved Pattern 1:", retrieved_pattern1)
print("Retrieved Pattern 2:", retrieved_pattern2)
