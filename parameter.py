import numpy as np

class Parameters:
    def __init__(self):
        self.weights = None
        self.bias = None

    def initialize(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
